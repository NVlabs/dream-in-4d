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
import os
import shutil
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *

@threestudio.register("zeroscope-system")
class Zeroscope(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        num_frames: int = 24
        guidance: dict = field(default_factory=dict)
        prompt_processor: dict = field(default_factory=dict)
        SD_view: int = 180

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def reshape_input(self, batch):
        batch_reshape = {}
        bs = batch["rays_o"].shape[0]
        T = batch["rays_o"].shape[1]

        for k, v in batch.items():
            if torch.is_tensor(v) and v.shape[0] == bs:
                batch_reshape[k] = v.reshape(bs * T, *v.shape[2:])
            else:
                batch_reshape[k] = v
        return batch_reshape

    def training_step(self, batch, batch_idx):
        loss_prefix = f"loss_"

        loss_terms = {}
        loss = 0.0

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value


        if len(batch["rays_o"].shape) == 5:
            bs = batch["rays_o"].shape[0]
            num_frames = batch["rays_o"].shape[1]
        else:
            bs = 1
            num_frames = batch["rays_o"].shape[0]

        def reshape_output(render_out):
            render_out_reshape = {}
            for k, v in render_out.items():
                if torch.is_tensor(v) and v.shape[0] == bs * num_frames:
                    render_out_reshape[k] = v.reshape(bs, num_frames, *v.shape[1:])
                else:
                    render_out_reshape[k] = v
            return render_out_reshape

        if len(batch["rays_o"].shape) == 5:
            batch = self.reshape_input(batch)
        out = self(batch)
        out = reshape_output(out)
        batch = reshape_output(batch)

        guidance_eval = (
            self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )
        guidance_out = self.guidance(
            out["comp_rgb"],
            **batch,
            prompt_utils=self.prompt_processor(),
            guidance_eval=guidance_eval,
            return_sds_color_loss=self.C(self.cfg.loss.lambda_sds_color) > 0
        )

        if guidance_eval:
            self.guidance_evaluation_save_video(
                out["comp_rgb"].detach(),
                guidance_out["eval"],
            )

        set_loss("sds_video", guidance_out["loss_sds"])
        if self.C(self.cfg.loss.lambda_sds_color) > 0:
            set_loss("sds_color", guidance_out["loss_sds_color"])


        if self.C(self.cfg.loss.lambda_deformation_spatial_reg) > 0:
            displacement = out["comp_displacement"]
            set_loss("deformation_spatial_reg",
                     (displacement[:, 1:] - displacement[:, :-1]).square().mean(0).sum() +
                     (displacement[:, :, 1:] - displacement[:, :, :-1]).square().mean(0).sum() +
                     (displacement[:, :, :, 1:] - displacement[:, :, :, :-1]).square().mean(0).sum())

        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss += loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log("train/loss", loss, prog_bar=True)
        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        if len(batch["rays_o"].shape) == 5:
            batch = self.reshape_input(batch)
        out = self(batch)

        if out["comp_rgb"].shape[-1] == 4:
            # BTHWC -> BCTHW -> BTHWC
            out["comp_rgb"] = self.guidance.decode_latents(out["comp_rgb"].unsqueeze(0).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)[0]
        bs = out["comp_rgb"].shape[0]
        for i in range(bs):
            self.save_image_grid(
                f"it{self.true_global_step}-val/{batch['index'][0] * bs + i}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][i],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][i],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][i],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][i],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][i, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                # claforte: TODO: don't hardcode the frame numbers to record... read them from cfg instead.
                name=f"validation_step_batchidx_{i}",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass
        filestem = f"it{self.true_global_step}-val"
        try:
            self.save_img_sequence(
                filestem,
                filestem,
                "(\d+)\.png",
                save_format="mp4",
                fps=24,
                name="validation_epoch_end",
                step=self.true_global_step,
            )
            if os.path.exists(os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")):
                try:
                    shutil.rmtree(
                        os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
                    )
                except:
                    print("somehow cannot remove tree")
        except:
            print("Probably, an image is corrupted")

    def test_step(self, batch, batch_idx):
        if len(batch["rays_o"].shape) == 5:
            batch = self.reshape_input(batch)
        out = self(batch)
        if out["comp_rgb"].shape[-1] == 4:
            # BTHWC -> BCTHW -> BTHWC
            out["comp_rgb"] = self.guidance.decode_latents(out["comp_rgb"].unsqueeze(0).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)[0]
        bs = out["comp_rgb"].shape[0]

        for i in range(bs):
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0] * bs + i}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][i],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][i],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][i],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][i],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                # + (
                #     [
                #         {
                #             "type": "rgb",
                #             "img": out["comp_displacement"][i],
                #             "kwargs": {"data_format": "HWC", "data_range": (-0.05, 0.05)},
                #         }
                #     ]
                #     if "comp_displacement" in out
                #     else []
                # )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][i, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=24,
            name="test",
            step=self.true_global_step,
        )
        if os.path.exists(os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")):
            try:
                shutil.rmtree(
                    os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")
                )
            except:
                print("somehow cannot remove tree")

    def guidance_evaluation_save_video(self, comp_rgb, guidance_eval_out, name=""):
        save_name = f"it{self.true_global_step}_target"
        self.save_results(guidance_eval_out["target_image"].permute(0, 2, 3, 4, 1)[:1].detach().cpu().numpy(), save_name=save_name)

        save_name = f"it{self.true_global_step}_generated"
        if comp_rgb.shape[-1] == 4:
            comp_rgb = self.guidance.decode_latents(comp_rgb.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        self.save_results(torch.clamp(comp_rgb, 0., 1.).permute(0, 1, 2, 3, 4)[:1].detach().cpu().numpy(), save_name=save_name)
