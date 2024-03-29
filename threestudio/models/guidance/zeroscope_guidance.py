# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/dream-in-4D/blob/main/LICENSE_nvidia
#
# Written by Yufeng Zheng
# ------------------------------------------------------------------------------
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, DiffusionPipeline

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

@threestudio.register("zeroscope-guidance")
class ZeroscopeGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "cerspense/zeroscope_v2_576w"
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None
        half_precision_weights: bool = True

        min_step_percent: Any = 0.02
        max_step_percent: Any = 0.98

        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = False

        use_hifa: bool = False
        num_hifa_steps: int = 4
        step_square_root: bool = False

        img_size: Any = 512

        guidance_cache_dir: str = "/root/.cache/huggingface/hub"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Zeroscope ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "torch_dtype": self.weights_dtype,
        }

        self.pipe = DiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
            cache_dir=self.cfg.guidance_cache_dir
        ).to(self.device)

        del self.pipe.text_encoder
        del self.pipe.scheduler
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler.set_timesteps(self.cfg.num_hifa_steps)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Zeroscope!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        sample = self.unet(
                latents.to(self.weights_dtype),
                t.to(self.weights_dtype),
                encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            )[0]
        sample = sample.to(input_dtype)

        return sample

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, video: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = video.dtype
        video = video * 2.0 - 1.0
        posterior = self.vae.encode(video.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(self, latents):
        input_dtype = latents.dtype

        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        video = self.vae.decode(latents.to(self.weights_dtype)).sample
        video = video.reshape((batch_size, num_frames, -1) + video.shape[2:]).permute(0, 2, 1, 3, 4)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video.to(input_dtype)
        return video

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 T 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        if self.cfg.view_dependent_prompting:
            # text embedding shape: [2 * bs, num_frames, 77, 1024]
            # We're choosing the text embedding of the first frame,
            # since video diffusion model only takes 1 take embedding per video
            text_embeddings = text_embeddings[:, 0]

        with torch.no_grad():
            input_latents = latents
            noise = torch.randn_like(latents)

            latents = self.scheduler.add_noise(latents, noise, t)

            if self.cfg.use_hifa:
                iter_t = t
                while iter_t[0] > 0:
                    prev_t = iter_t - self.num_train_timesteps // self.cfg.num_hifa_steps
                    # pred noise
                    latent_model_input = torch.cat([latents] * 2, dim=0)

                    noise_pred = self.forward_unet(
                        latent_model_input,
                        torch.cat([iter_t] * 2),
                        encoder_hidden_states=text_embeddings
                    )

                    # perform guidance (high scale from paper!)
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_text + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # reshape latents
                    bsz, channel, frames, width, height = latents.shape
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                    noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                    # compute the previous noisy sample x_t -> x_t-1
                    # using iter_t[0] because self.scheduler.step only accept float as the timestep,
                    # So the timestep for the entire batch MUST be the same.
                    # Either do not use any randomness in HiFA timestep schedule,
                    # or use batch size = 1
                    latents = self.scheduler.step(noise_pred, iter_t[0].cpu().numpy(), latents, eta=1.0).prev_sample

                    # reshape latents back
                    latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                    iter_t = prev_t
                grad = input_latents - latents
            else:
                latent_model_input = torch.cat([latents] * 2, dim=0)

                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings
                )

                # perform guidance (high scale from paper!)
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                if self.cfg.weighting_strategy == "sds":
                    # w(t), sigma_t^2
                    w = (1 - self.alphas[t]).view(-1, 1, 1, 1, 1)
                elif self.cfg.weighting_strategy == "uniform":
                    w = 1
                elif self.cfg.weighting_strategy == "fantasia3d":
                    w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1, 1)
                else:
                    raise ValueError(
                        f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
                    )
                grad = w * (noise_pred - noise)

        guidance_eval_utils = {
            "target": input_latents - grad
        }

        return grad, guidance_eval_utils

    def __call__(
        self,
        rgb: Float[Tensor, "B T H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_eval=False,
        return_sds_color_loss=False,
        **kwargs,
    ):
        batch_size, num_frames, height, width, channels = rgb.shape

        latents: Float[Tensor, "B 4 T 64 64"]
        rgb_BCHW = rgb.permute(0, 1, 4, 2, 3).view(batch_size * num_frames, channels, height, width)

        rgb_as_latents = (channels == 4)

        if rgb_as_latents:
            latents = F.interpolate(rgb_BCHW, (int(self.cfg.img_size[0] / 8), int(self.cfg.img_size[1] / 8)), mode="bilinear", align_corners=False)
        else:
            rgb_BCHW_512 = F.interpolate(rgb_BCHW, (self.cfg.img_size[0], self.cfg.img_size[1]), mode="bilinear", align_corners=False)
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        latents = latents.reshape((batch_size, num_frames, -1) + latents.shape[2:]).permute(0, 2, 1, 3, 4) # B C T H W

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        grad, guidance_eval_utils = self.compute_grad_sds(
            latents, t, prompt_utils, elevation, azimuth, camera_distances)

        grad = torch.nan_to_num(grad)

        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }
        if self.cfg.use_hifa and return_sds_color_loss:
            loss_sds_color = F.mse_loss(rgb_BCHW_512.reshape(batch_size, num_frames, channels, self.cfg.img_size[0], self.cfg.img_size[1]).permute(0, 2, 1, 3, 4),
                                        self.decode_latents(target).detach(), reduction="sum") / batch_size
            guidance_out["loss_sds_color"] = loss_sds_color
        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        target
    ):
        target_image = self.decode_latents(target)

        return {
            "target_image": target_image,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)


        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

