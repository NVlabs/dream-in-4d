# ------------------------------------------------------------------------------
# Code is modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/systems/zero123.py
# To view a copy of this license, visit
# https://github.com/threestudio-project/threestudio/blob/main/LICENSE
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/dream-in-4D/blob/main/LICENSE_nvidia
#
# Modified by Yufeng Zheng
# ------------------------------------------------------------------------------
import os
import random
import shutil
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("magic123-new-system")
class Magic123(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        freq: dict = field(default_factory=dict)
        refinement: bool = False

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.guidance_2D = threestudio.find(self.cfg.guidance_type[1])(self.cfg.guidance.guidance_2D)
        self.guidance_3D = threestudio.find(self.cfg.guidance_type[0])(self.cfg.guidance.guidance_3D)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "guidance"
        """
        if guidance == "ref":
            ambient_ratio = 1.0
            shading = "diffuse"
            batch["shading"] = shading
        elif guidance == "guidance":
            batch = batch["random_camera"]
            ambient_ratio = 0.1 + 0.9 * random.random()

        batch["bg_color"] = None  # manually composite white background for zero123
        batch["ambient_ratio"] = ambient_ratio

        out = self(batch)

        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "guidance"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0 and self.true_global_step > 0
        )

        if guidance == "ref":
            gt_mask = batch["mask"]
            gt_rgb = batch["rgb"]
            # color loss
            gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                1 - gt_mask.float()
            )
            set_loss("rgb", F.mse_loss(gt_rgb, out["comp_rgb"]))

            # mask loss
            set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

            # depth loss
            if self.C(self.cfg.loss.lambda_depth) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                with torch.no_grad():
                    A = torch.cat(
                        [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                    )  # [B, 2]
                    X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                    valid_gt_depth = A @ X  # [B, 1]
                set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

            # relative depth loss
            if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                valid_pred_depth = out["depth"][gt_mask]  # [B,]
                set_loss(
                    "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * batch["ref_normal"][gt_mask.squeeze(-1)]
                )  # [B, 3]
                valid_pred_normal = (
                    2 * out["comp_normal"][gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )
        elif guidance == "guidance":
            # zero123
            comp_rgb_w_white_bg = out["comp_rgb_fg"] + (1.0 - out["opacity"])
            guidance_out_3D = self.guidance_3D(
                comp_rgb_w_white_bg,
                **batch,
                guidance_eval=guidance_eval,
            )

            set_loss("sds_3d", guidance_out_3D["loss_sds"])

            # deepfloyd
            guidance_out_2D = self.guidance_2D(
                out["comp_rgb"],
                self.prompt_utils,
                **batch,
                guidance_eval=guidance_eval,
            )

            set_loss("sds_2d", guidance_out_2D["loss_sds"])

        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )
        else:
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())

        loss = 0.0
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

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            comp_rgb_w_white_bg = out["comp_rgb_fg"] + (1.0 - out["opacity"])
            self.guidance_evaluation_save(
                comp_rgb_w_white_bg.detach()[: guidance_out_3D["eval"]["bs"]],
                guidance_out_3D["eval"], name='3D'
            )
            self.guidance_evaluation_save(
                    out["comp_rgb"].detach()[: guidance_out_2D["eval"]["bs"]],
                    guidance_out_2D["eval"], name='2D'
            )

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        do_ref = True
        do_guidance = True

        total_loss = 0.0
        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance")
            total_loss += out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref")
            total_loss += out["loss"]

        self.log("train/loss", total_loss, prog_bar=True)
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
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
                        "img": torch.clamp((out["depth"][0] - 2) / (6 - 2), 0, 1),
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name=f"validation_step_batchidx_{batch_idx}"
            if batch_idx in [0, 7, 15, 23, 29]
            else None,
            step=self.true_global_step,
        )


    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"
        self.save_img_sequence(
            filestem,
            filestem,
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="validation_epoch_end",
            step=self.true_global_step,
        )
        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
        )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
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
                        "img": torch.clamp((out["depth"][0] - 2) / (6 - 2), 0, 1),
                        "kwargs": {},
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
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
            fps=30,
            name="test",
            step=self.true_global_step,
        )

        shutil.rmtree(
            os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test")
        )