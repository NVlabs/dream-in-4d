# ------------------------------------------------------------------------------
# Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/dream-in-4D/blob/main/LICENSE_nvidia
#
# Written by Yufeng Zheng
# ------------------------------------------------------------------------------
import bisect
import math
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *




@dataclass
class RandomCameraTrajectoryDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    num_frames: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    n_split_val: int = 4
    n_split_test: int = 96
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    delta_percentage: float = 0.2  # camera trajectory
    trajectory_percentage: float = 0.2  # camera trajectory
    delta_time: float = 0  # time sampling
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    eval_azimuth_deg: float = 0.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0
    relative_radius: bool = True
    rays_d_normalize: bool = True


class RandomCameraTrajectoryIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: RandomCameraTrajectoryDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.num_frames: List[int] = (
            [self.cfg.num_frames]
            if isinstance(self.cfg.num_frames, int)
            else self.cfg.num_frames
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes) == len(self.num_frames)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.num_frame: int = self.num_frames[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range




    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.num_frame = self.num_frames[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}, num_frame: {self.num_frame}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]

    def sample_delta(self, range, value, percentage):
        min_value = torch.clamp(range[0] * torch.ones_like(value) - value, min=-percentage * (range[1] - range[0]))
        max_value = torch.clamp(range[1] * torch.ones_like(value) - value, max=percentage * (range[1] - range[0]))
        delta = torch.rand(self.batch_size) * (max_value - min_value) + min_value
        return delta

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = ((torch.rand(self.batch_size) + torch.arange(self.batch_size)) / self.batch_size
                           * (self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0])
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )
        if random.random() < self.cfg.trajectory_percentage:
            elevation_delta = self.sample_delta(range=self.elevation_range, value=elevation_deg, percentage=self.cfg.delta_percentage / 4)
            azimuth_delta = self.sample_delta(range=self.azimuth_range, value=azimuth_deg, percentage=self.cfg.delta_percentage)
            camera_distance_delta = self.sample_delta(range=self.camera_distance_range, value=camera_distances, percentage=self.cfg.delta_percentage)

            elevation_deg = elevation_deg.unsqueeze(1).expand(-1, self.num_frame) + torch.linspace(0, 1, self.num_frame)[None, :] * elevation_delta[:, None]
            azimuth_deg = azimuth_deg.unsqueeze(1).expand(-1, self.num_frame) + torch.linspace(0, 1, self.num_frame)[None, :] * azimuth_delta[:, None]
            camera_distances = camera_distances.unsqueeze(1).expand(-1, self.num_frame) + torch.linspace(0, 1, self.num_frame)[None, :] * camera_distance_delta[:, None]
        else:
            elevation_deg = elevation_deg.unsqueeze(1).expand(-1, self.num_frame)
            azimuth_deg = azimuth_deg.unsqueeze(1).expand(-1, self.num_frame)
            camera_distances = camera_distances.unsqueeze(1).expand(-1, self.num_frame)

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B T"] = (
            torch.rand(self.batch_size)[:, None].expand(self.batch_size, self.num_frame) * (
                self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )  # same fovy for each video, camera distance is changing already, probably no need to change fovy
        fovy = fovy_deg * math.pi / 180

        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180


        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B T 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B T 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B T 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, None, :].repeat(self.batch_size, self.num_frame, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B T 3"] = (
            torch.rand(self.batch_size, 3)[:, None, :].expand(self.batch_size, self.num_frame, 3)
            * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B T 3"] = (
            torch.randn(self.batch_size, 3)[:, None, :].expand(self.batch_size, self.num_frame, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B T 3"] = (
            torch.randn(self.batch_size, 3)[:, None, :].expand(self.batch_size, self.num_frame, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )  # light distance is kept fixed

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions[:, 0]
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions[:, 0], dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi - 2 * math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        # light position is fixed for each video
        light_positions = light_positions[:, None, :].expand(self.batch_size, self.num_frame, 3)

        lookat: Float[Tensor, "B T 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B T 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B T 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B T 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :, :1])], dim=-2
        )
        c2w[:, :, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B T"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B T H W 3"] = self.directions_unit_focal[None, None, :, :, :].repeat(self.batch_size, self.num_frame, 1, 1, 1)
        directions[:, :, :, :, :2] = (
            directions[:, :, :, :, :2] / focal_length[:, :, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize)

        proj_mtx: Float[Tensor, "B T 4 4"] = (get_projection_matrix(fovy.reshape(self.batch_size * self.num_frame),
                                                                   self.width / self.height, 0.1, 1000.0).reshape(self.batch_size, self.num_frame, 4, 4)) # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B T 4 4"] = get_mvp_matrix(c2w.reshape(self.batch_size * self.num_frame, 4, 4),
                                                           proj_mtx.reshape(self.batch_size * self.num_frame, 4, 4)).reshape(self.batch_size, self.num_frame, 4, 4)

        # time sampling
        time_length = torch.rand(self.batch_size, 1) * self.cfg.delta_time + (1 - self.cfg.delta_time) # delta_time = 0.2 --> time_length in [0.8, 1.0]
        time_start = torch.rand(self.batch_size, 1) * (1 - time_length)  # time_length = 0.9, time_start in [0, 0.1]
        times = torch.linspace(0, 1, self.num_frame)[None, :].expand(self.batch_size, self.num_frame)
        times = times * time_length + time_start
        batch = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "times": times,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
        }

        return batch


class RandomCameraTrajectoryDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraTrajectoryDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.num_frame = self.cfg.n_val_views
        else:
            self.num_frame = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            n_splits = self.cfg.n_split_val
            if self.num_frame % n_splits != 0:
                self.num_frame = self.num_frame // n_splits * n_splits
            list = []
            for i in range(n_splits):
                list += torch.linspace(360 / n_splits * i, 360 / n_splits * i, self.num_frame // n_splits)

            azimuth_deg = self.cfg.eval_azimuth_deg + torch.stack(list)
            self.times = torch.linspace(0, 1, self.num_frame // n_splits).repeat(n_splits)

        else:
            n_splits = self.cfg.n_split_test

            assert n_splits % 2 == 0
            assert self.num_frame % n_splits == 0
            if self.num_frame % n_splits != 0:
                self.num_frame = self.num_frame // n_splits * n_splits
            list = []
            for i in range(n_splits):
                if i % 2 == 0:
                    list += torch.linspace(360 / n_splits * i, 360 / n_splits * i, self.num_frame // n_splits)
                else:
                    list += torch.linspace(360 / n_splits * (i - 1), 360 / n_splits * (i + 1), self.num_frame // n_splits)

            azimuth_deg = self.cfg.eval_azimuth_deg + torch.stack(list)
            self.times = torch.linspace(0, 1, self.num_frame // n_splits).repeat(n_splits)

        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.num_frame, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.num_frame, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=self.cfg.rays_d_normalize)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.num_frame

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
            "times": self.times[index]
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch

@register("random-camera-trajectory-datamodule")
class RandomCameraTrajectoryDataModule(pl.LightningDataModule):
    cfg: RandomCameraTrajectoryDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraTrajectoryDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomCameraTrajectoryIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraTrajectoryDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraTrajectoryDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
