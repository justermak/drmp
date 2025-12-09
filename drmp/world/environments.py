from abc import ABC
from typing import Any, Dict, Tuple

import numpy as np
import torch

from drmp.config import N_DIM
from drmp.utils.torch_timer import TimerCUDA
from drmp.utils.trajectory_utils import interpolate_trajectories
from drmp.world.grid_map_sdf import GridMapSDF
from drmp.world.primitives import MultiBoxField, MultiSphereField, ObjectField
from drmp.world.robot import Robot


def get_envs():
    return {
        "EnvSimple2D": EnvSimple2D,
        "EnvDense2D": EnvDense2D,
    }


def create_workspace_boundary_boxes(limits: np.ndarray, r: float = 0.2) -> list:
    x_min, y_min = limits[0]
    x_max, y_max = limits[1]
    height = y_max - y_min
    width = x_max - x_min

    centers = np.array(
        [
            [x_min - r, (y_min + y_max) / 2],
            [x_max + r, (y_min + y_max) / 2],
            [(x_min + x_max) / 2, y_min - r],
            [(x_min + x_max) / 2, y_max + r],
        ]
    )

    half_sizes = np.array(
        [
            [r, height / 2 + r],
            [r, height / 2 + r],
            [width / 2 + r, r],
            [width / 2 + r, r],
        ]
    )

    return centers, half_sizes


class EnvBase(ABC):
    def __init__(
        self,
        limits: torch.Tensor,
        obj_field_fixed: ObjectField,
        obj_field_extra: ObjectField,
        sdf_cell_size: float,
        tensor_args: Dict[str, Any],
    ):
        self.tensor_args = tensor_args
        self.limits = limits
        self.limits_np = limits.cpu().numpy()

        self.obj_field_fixed = obj_field_fixed
        self.obj_field_extra = obj_field_extra

        with TimerCUDA() as t:
            self.grid_map_sdf_fixed = GridMapSDF(
                self.limits,
                sdf_cell_size,
                self.obj_field_fixed,
                tensor_args=self.tensor_args,
            )
            self.grid_map_sdf_extra = GridMapSDF(
                self.limits,
                sdf_cell_size,
                self.obj_field_extra,
                tensor_args=self.tensor_args,
            )
            print(f"Precomputing the SDF grid and gradients took: {t.elapsed:.3f} sec")

        self.q_distribution = torch.distributions.uniform.Uniform(
            self.limits[0], self.limits[1]
        )

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "EnvBase":
        if device is not None:
            self.tensor_args["device"] = device
        if dtype is not None:
            self.tensor_args["dtype"] = dtype
        self.limits = self.limits.to(device=device, dtype=dtype)
        self.obj_field_fixed.to(device=device, dtype=dtype)
        self.obj_field_extra.to(device=device, dtype=dtype)
        self.grid_map_sdf_fixed.to(device=device, dtype=dtype)
        self.grid_map_sdf_extra.to(device=device, dtype=dtype)
        # Recreate distribution on new device
        self.q_distribution = torch.distributions.uniform.Uniform(
            self.limits[0], self.limits[1]
        )
        return self

    def random_q(self, shape) -> torch.Tensor:
        return self.q_distribution.sample(shape)

    def get_collision_mask(
        self,
        robot: Robot,
        qs: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        qs = robot.get_position(qs)
        collision_mask_fixed = torch.zeros(
            qs.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        collision_mask_extra = torch.zeros(
            qs.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        if on_fixed:
            sdf_fixed = self.grid_map_sdf_fixed.compute_approx_signed_distance(qs)
            collision_mask_fixed = (sdf_fixed < robot.margin).any(dim=-1)
        if on_extra:
            sdf_extra = self.grid_map_sdf_extra.compute_approx_signed_distance(qs)
            collision_mask_extra = (sdf_extra < robot.margin).any(dim=-1)

        collision_mask = collision_mask_fixed | collision_mask_extra
        return collision_mask

    def compute_cost(
        self,
        qs: torch.Tensor,
        robot: Robot,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        qs = robot.get_position(qs)
        total_cost = torch.zeros(qs.shape[:-1], **self.tensor_args)
        if on_fixed:
            sdf_fixed = self.grid_map_sdf_fixed.compute_approx_signed_distance(qs)
            cost_fixed = torch.relu(robot.margin - sdf_fixed).sum(dim=-1)
            total_cost += cost_fixed
        if on_extra:
            sdf_extra = self.grid_map_sdf_extra.compute_approx_signed_distance(qs)
            cost_extra = torch.relu(robot.margin - sdf_extra).sum(dim=-1)
            total_cost += cost_extra

        return total_cost

    def random_collision_free_q(
        self, robot: Robot, n_samples: int, batch_size=100000, max_tries=1000
    ) -> torch.Tensor:
        samples = torch.zeros((n_samples, N_DIM), **self.tensor_args)
        cur = 0
        for i in range(max_tries):
            qs = self.random_q((batch_size,))
            collision_mask = self.get_collision_mask(robot=robot, qs=qs).squeeze()
            n = torch.sum(~collision_mask).item()
            n = min(n, n_samples - cur)
            samples[cur : cur + n] = qs[~collision_mask][:n]
            cur += n
            if cur >= n_samples:
                break

        return samples.squeeze(), cur >= n_samples

    def random_collision_free_start_goal(
        self,
        robot: Robot,
        n_samples: int,
        threshold_start_goal_pos,
        batch_size=100000,
        max_tries=1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        samples_start = torch.zeros((n_samples, N_DIM), **self.tensor_args)
        samples_goal = torch.zeros((n_samples, N_DIM), **self.tensor_args)
        cur = 0
        for _ in range(max_tries):
            qs, success = self.random_collision_free_q(
                robot=robot,
                n_samples=n_samples * 2,
                batch_size=batch_size,
                max_tries=max_tries,
            )
            if not success:
                return None, None, False
            start_state_pos, goal_state_pos = qs[:n_samples], qs[n_samples:]
            threshold_mask = (
                torch.linalg.norm(start_state_pos - goal_state_pos, dim=-1)
                > threshold_start_goal_pos
            )
            n = torch.sum(threshold_mask).item()
            n = min(n, n_samples - cur)
            samples_start[cur : cur + n] = start_state_pos[threshold_mask][:n]
            samples_goal[cur : cur + n] = goal_state_pos[threshold_mask][:n]
            cur += n
            if cur >= n_samples:
                break

        return samples_start, samples_goal, cur >= n_samples

    def get_trajectories_collision_and_free(
        self,
        trajectories: torch.Tensor,
        robot: Robot,
        n_interpolate: int = 5,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectories_interpolated = interpolate_trajectories(trajectories=trajectories, n_interpolate=n_interpolate)
        points_collision_mask = self.get_collision_mask(
            robot, trajectories_interpolated, on_fixed=on_fixed, on_extra=on_extra
        )
        trajectories_collision_mask = points_collision_mask.any(dim=-1)
        trajectories_collision = trajectories[trajectories_collision_mask]
        trajectories_free = trajectories[~trajectories_collision_mask]

        return trajectories_collision, trajectories_free, points_collision_mask


class EnvSimple2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
    ):
        # Create workspace boundary boxes
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiSphereField(
                    centers=np.array(
                        [
                            [-0.43378472328186035, 0.3334643840789795],
                            [0.3313474655151367, 0.6288051009178162],
                            [-0.5656964778900146, -0.484994500875473],
                            [0.42124247550964355, -0.6656165719032288],
                            [0.05636655166745186, -0.5149664282798767],
                            [-0.36961784958839417, -0.12315540760755539],
                            [-0.8740217089653015, -0.4034936726093292],
                            [-0.6359214186668396, 0.6683124899864197],
                            [0.808782160282135, 0.5287870168685913],
                            [-0.023786112666130066, 0.4590069353580475],
                            [0.1455741971731186, 0.16420497000217438],
                            [0.628413736820221, -0.43461447954177856],
                            [0.17965620756149292, -0.8926276564598083],
                            [0.6775968670845032, 0.8817358016967773],
                            [-0.3608766794204712, 0.8313458561897278],
                        ]
                    ),
                    radii=np.array(
                        [
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    boundary_centers,
                    boundary_half_sizes,
                    tensor_args=tensor_args,
                ),
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.15, 0.15],
                            [-0.075, -0.85],
                            [-0.1, -0.1],
                            [0.45, -0.1],
                            [0.5, 0.35],
                            [-0.6, -0.85],
                            [0.05, 0.85],
                            [-0.8, 0.15],
                            [0.8, -0.8],
                        ]
                    ),
                    np.array(
                        [
                            0.05,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                            0.1,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    np.array(
                        [
                            [0.45, -0.1],
                            [-0.25, -0.5],
                            [0.8, 0.1],
                        ]
                    ),
                    np.array(
                        [
                            [0.2, 0.2],
                            [0.15, 0.15],
                            [0.15, 0.15],
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            tensor_args=tensor_args,
        )


class EnvDense2D(EnvBase):
    def __init__(
        self,
        tensor_args: Dict[str, Any],
    ):
        limits_np = np.array([[-1.0, -1.0], [1.0, 1.0]])
        boundary_centers, boundary_half_sizes = create_workspace_boundary_boxes(
            limits_np
        )

        obj_field_fixed = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.43378472328186035, 0.3334643840789795],
                            [0.3313474655151367, 0.6288051009178162],
                            [-0.5656964778900146, -0.484994500875473],
                            [0.42124247550964355, -0.6656165719032288],
                            [0.05636655166745186, -0.5149664282798767],
                            [-0.36961784958839417, -0.12315540760755539],
                            [-0.8740217089653015, -0.4034936726093292],
                            [-0.6359214186668396, 0.6683124899864197],
                            [0.808782160282135, 0.5287870168685913],
                            [-0.023786112666130066, 0.4590069353580475],
                            [0.11544948071241379, -0.12676022946834564],
                            [0.1455741971731186, 0.16420497000217438],
                            [0.628413736820221, -0.43461447954177856],
                            [0.17965620756149292, -0.8926276564598083],
                            [0.6775968670845032, 0.8817358016967773],
                            [-0.3608766794204712, 0.8313458561897278],
                        ]
                    ),
                    np.array(
                        [
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                            0.125,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    centers=np.array(
                        [
                            [0.607781708240509, 0.19512386620044708],
                            [0.5575312972068787, 0.5508843064308167],
                            [-0.3352295458316803, -0.6887519359588623],
                            [-0.6572632193565369, 0.31827881932258606],
                            [-0.664594292640686, -0.016457155346870422],
                            [0.8165988922119141, -0.19856023788452148],
                            [-0.8222246170043945, -0.6448580026626587],
                            [-0.2855989933013916, -0.36841487884521484],
                            [-0.8946458101272583, 0.8962447643280029],
                            [-0.23994405567646027, 0.6021060943603516],
                            [-0.006193588487803936, 0.8456171751022339],
                            [0.305103600025177, -0.3661990463733673],
                            [-0.10704007744789124, 0.1318950206041336],
                            [0.7156378626823425, -0.6923345923423767],
                            *boundary_centers,
                        ]
                    ),
                    half_sizes=np.array(
                        [
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            [0.1, 0.1],
                            *boundary_half_sizes,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        obj_field_extra = ObjectField(
            [
                MultiSphereField(
                    np.array(
                        [
                            [-0.4, 0.1],
                            [-0.075, -0.85],
                            [-0.1, -0.1],
                        ]
                    ),
                    np.array(
                        [
                            0.075,
                            0.1,
                            0.075,
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
                MultiBoxField(
                    np.array(
                        [
                            [0.45, -0.1],
                            [0.35, 0.35],
                            [-0.6, -0.85],
                            [-0.65, -0.25],
                        ]
                    ),
                    np.array(
                        [
                            [0.1, 0.1],
                            [0.05, 0.075],
                            [0.05, 0.125],
                            [0.075, 0.05],
                        ]
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        )

        super().__init__(
            limits=torch.tensor(limits_np, **tensor_args),
            obj_field_fixed=obj_field_fixed,
            obj_field_extra=obj_field_extra,
            sdf_cell_size=0.005,
            tensor_args=tensor_args,
        )
