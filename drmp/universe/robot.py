from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from drmp.universe.environments import EnvBase
from drmp.utils import (
    fit_bsplines_to_trajectories,
    get_trajectories_from_bsplines,
    interpolate_trajectories,
)


class RobotBase(ABC):
    def __init__(
        self,
        margin: float,
        dt: float,
        spline_degree: int,
        tensor_args: Dict[str, Any],
    ) -> None:
        self.tensor_args = tensor_args
        self.margin = margin
        self.dt = dt
        self.spline_degree = spline_degree
        self.n_dim: int = None

    @abstractmethod
    def enforce_rigid_constraints(self, points: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_collision_points(self, points: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def generate_random_points(self, env: EnvBase, n_samples: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_anchor_points(self, points: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def create_straight_line_trajectory(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def smoothen_trajectory(
        self,
        trajectory: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        pass

    def get_position(
        self,
        trajectories: torch.Tensor,
    ) -> torch.Tensor:
        return trajectories[..., : self.n_dim]

    def get_position_interpolated(
        self,
        control_points: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        control_points_pos = self.get_position(control_points)
        trajectories_pos = get_trajectories_from_bsplines(
            control_points=control_points_pos,
            n_support_points=n_support_points,
            degree=self.spline_degree,
        )
        trajectories_constrained = self.enforce_rigid_constraints(trajectories_pos)
        return trajectories_constrained

    def fit_bsplines_to_trajectories(
        self,
        trajectories: torch.Tensor,
        n_control_points: int,
    ) -> torch.Tensor:
        trajectories_pos = self.get_position(trajectories)
        control_points_pos = fit_bsplines_to_trajectories(
            trajectories=trajectories_pos,
            n_control_points=n_control_points,
            degree=self.spline_degree,
        )
        return control_points_pos

    def get_velocity(
        self,
        trajectories: torch.Tensor,
        mode: str = None,
        dt: float = None,
    ) -> torch.Tensor:
        dt = dt if dt is not None else self.dt
        if mode == "forward":
            trajectories_pos = self.get_position(trajectories)
            trajectories_vel = torch.diff(trajectories_pos, dim=-2) / dt
            trajectories_vel = torch.cat(
                [trajectories_vel, torch.zeros_like(trajectories_vel[:, :1, :])], dim=-2
            )
            
        elif mode == "central":
            trajectories_pos = self.get_position(trajectories)
            trajectories_vel = (
                trajectories_pos[:, 2:, :] - trajectories_pos[:, :-2, :]
            ) / (2 * dt)
            trajectories_vel = torch.cat(
                [
                    torch.zeros_like(trajectories_vel[:, :1, :]),
                    trajectories_vel,
                    torch.zeros_like(trajectories_vel[:, :1, :]),
                ],
                dim=-2,
            )
            
        elif mode == "avg":
            trajectories_pos = self.get_position(trajectories)
            displacement = trajectories_pos[:, -1, :] - trajectories_pos[:, 0, :]
            avg_vel = displacement / (dt * (trajectories_pos.shape[1] - 2))
            trajectories_vel = torch.cat(
                [
                    torch.zeros_like(avg_vel).unsqueeze(1),
                    avg_vel.unsqueeze(1).repeat(1, trajectories_pos.shape[1] - 2, 1),
                    torch.zeros_like(avg_vel).unsqueeze(1),
                ],
                dim=1,
            )
            
        elif trajectories.shape[-1] >= 2 * self.n_dim:
            trajectories_vel = trajectories[..., self.n_dim : 2 * self.n_dim]
            
        else:
            trajectories_vel = torch.zeros_like(trajectories[..., : self.n_dim])
            
        return trajectories_vel

    def get_acceleration(
        self,
        trajectories: torch.Tensor,
        mode: str,
        dt: float = None,
    ) -> torch.Tensor:
        dt = dt if dt is not None else self.dt
        if mode == "forward":
            trajectories_vel = self.get_velocity(trajectories, mode="forward", dt=dt)
            trajectories_acc = self.get_velocity(trajectories_vel, mode="forward", dt=dt)
            
        elif mode == "central":
            trajectories_vel = self.get_velocity(trajectories, mode="central", dt=dt)
            trajectories_acc = self.get_velocity(trajectories_vel, mode="central", dt=dt)
            
        else:
            trajectories_acc = torch.zeros_like(trajectories[..., : self.n_dim])
            
        return trajectories_acc
    
    def get_jerk(self, trajectories: torch.Tensor, mode: str = "forward", dt: float = None) -> torch.Tensor:
        dt = dt if dt is not None else self.dt
        if mode == "forward":
            trajectories_acc = self.get_acceleration(trajectories, mode="forward", dt=dt)
            trajectories_jerk = self.get_velocity(trajectories_acc, mode="forward", dt=dt)
            
        elif mode == "central":
            trajectories_acc = self.get_acceleration(trajectories, mode="central", dt=dt)
            trajectories_jerk = self.get_velocity(trajectories_acc, mode="central", dt=dt)
            
        else:
            trajectories_jerk = torch.zeros_like(trajectories[..., : self.n_dim])
            
        return trajectories_jerk

    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.shape[-1] == self.n_dim:
            trajectories_inverted = torch.flip(trajectories, dims=[-2])

        elif trajectories.shape[-1] >= 2 * self.n_dim:
            trajectories_reversed = torch.flip(trajectories, dims=[-2])
            trajectories_pos_reversed = self.get_position(trajectories_reversed)
            trajectories_vel_reversed = -self.get_velocity(trajectories_reversed)
            trajectories_inverted = torch.cat(
                [trajectories_pos_reversed, trajectories_vel_reversed], dim=-1
            )
            
        else:
            raise ValueError(
                "Input tensor must have either only position or at least position and velocity concatenated."
            )
            
        return trajectories_inverted

    def get_collision_mask(
        self,
        env: EnvBase,
        points: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        points_pos = self.get_position(points)
        collision_points = self.get_collision_points(points_pos)
        
        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(
                collision_points
            )
            collision_mask_fixed = (sdf_fixed < self.margin).any(dim=[-1, -2])
        
        else:
            collision_mask_fixed = torch.zeros(
                collision_points.shape[:-2], dtype=torch.bool, device=self.tensor_args["device"]
            )

        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(
                collision_points
            )
            collision_mask_extra = (sdf_extra < self.margin).any(dim=[-1, -2])
            
        else:
            collision_mask_extra = torch.zeros(
                collision_points.shape[:-2], dtype=torch.bool, device=self.tensor_args["device"]
            )

        collision_mask = collision_mask_fixed | collision_mask_extra
        
        return collision_mask

    def compute_cost(
        self,
        env: EnvBase,
        trajectories: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        trajectories_pos = self.get_position(trajectories)
        collision_points = self.get_collision_points(trajectories_pos)

        total_cost = torch.zeros(collision_points.shape[:-2], **self.tensor_args)

        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(
                collision_points
            )
            cost_fixed = torch.relu(self.margin - sdf_fixed).sum(dim=[-1, -2])
            total_cost += cost_fixed

        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(
                collision_points
            )
            cost_extra = torch.relu(self.margin - sdf_extra).sum(dim=[-1, -2])
            total_cost += cost_extra

        return total_cost

    def random_collision_free_points(
        self,
        env: EnvBase,
        n_samples: int,
        use_extra_objects: bool = False,
        batch_size=100000,
        max_tries=1000,
    ) -> Tuple[torch.Tensor, bool]:
        samples = torch.zeros((n_samples, self.n_dim), **self.tensor_args)
        cur = 0
        for i in range(max_tries):
            points = self.generate_random_points(env, batch_size)

            collision_mask = self.get_collision_mask(
                env=env, points=points, on_extra=use_extra_objects
            ).squeeze()

            n = torch.sum(~collision_mask).item()
            n = min(n, n_samples - cur)

            if n > 0:
                samples[cur : cur + n] =  points[~collision_mask][:n]
                cur += n

            if cur >= n_samples:
                break

        return samples, cur >= n_samples

    def random_collision_free_start_goal(
        self,
        env: EnvBase,
        n_samples: int,
        threshold_start_goal_pos: float,
        use_extra_objects: bool = False,
        batch_size: int = 100000,
        max_tries: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        samples_start = torch.zeros((n_samples, self.n_dim), **self.tensor_args)
        samples_goal = torch.zeros((n_samples, self.n_dim), **self.tensor_args)
        cur = 0
        for _ in range(max_tries):
            points, success = self.random_collision_free_points(
                env=env,
                n_samples=n_samples * 2,
                use_extra_objects=use_extra_objects,
                batch_size=batch_size,
                max_tries=max_tries,
            )

            if not success:
                return None, None, False

            start, goal = points[:n_samples], points[n_samples:]
            start_anchors = self.get_anchor_points(start)
            goal_anchors = self.get_anchor_points(goal)
            threshold_mask = (
                torch.linalg.norm(start_anchors - goal_anchors, dim=-1)
                > threshold_start_goal_pos
            )
            
            n = torch.sum(threshold_mask).item()
            n = min(n, n_samples - cur)
            
            if n > 0:
                samples_start[cur : cur + n] = start[threshold_mask][:n]
                samples_goal[cur : cur + n] = goal[threshold_mask][:n]
                cur += n

            if cur >= n_samples:
                break

        return samples_start, samples_goal, cur >= n_samples

    def get_trajectories_collision_and_free(
        self,
        env: EnvBase,
        trajectories: torch.Tensor,
        n_interpolate: int = 5,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectories_interpolated = interpolate_trajectories(
            trajectories=trajectories, n_interpolate=n_interpolate
        )
        points_collision_mask = self.get_collision_mask(
            env, trajectories_interpolated, on_fixed=on_fixed, on_extra=on_extra
        )
        trajectories_collision_mask = points_collision_mask.any(dim=-1)
        trajectories_collision = trajectories[trajectories_collision_mask]
        trajectories_free = trajectories[~trajectories_collision_mask]

        return trajectories_collision, trajectories_free, points_collision_mask

    def gradient_step(
        self, trajectories: torch.Tensor, grad: torch.Tensor
    ) -> torch.Tensor:
        new_trajectories = trajectories + grad
        constrained_trajectories = self.enforce_rigid_constraints(new_trajectories)
        return constrained_trajectories


class RobotSphere2D(RobotBase):
    def __init__(
        self,
        margin: float,
        dt: float,
        spline_degree: int,
        tensor_args: Dict[str, Any],
    ) -> None:
        super().__init__(margin, dt, spline_degree, tensor_args)
        self.n_dim = 2

    def enforce_rigid_constraints(self, trajectories):
        return trajectories

    def get_collision_points(self, points: torch.Tensor) -> torch.Tensor:
        return points.unsqueeze(-2)

    def generate_random_points(self, env: EnvBase, n_samples: int) -> torch.Tensor:
        return env.random_points((n_samples,))

    def get_anchor_points(self, points):
        return points

    def create_straight_line_trajectory(
        self,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        alphas = torch.linspace(0, 1, n_support_points, **self.tensor_args)
        pos = start_pos.unsqueeze(0) + (goal_pos - start_pos).unsqueeze(
            0
        ) * alphas.unsqueeze(1)

        return pos

    def smoothen_trajectory(
        self,
        trajectory: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        trajectory_augmented = torch.cat(
            [trajectory[:1, :], trajectory, trajectory[-1:, :]], dim=0
        )
        pos = get_trajectories_from_bsplines(
            control_points=trajectory_augmented,
            n_support_points=n_support_points,
            degree=self.spline_degree,
        )
        
        return pos


class RobotL2D(RobotBase):
    def __init__(
        self,
        margin: float,
        dt: float,
        spline_degree: int,
        width: float,
        height: float,
        n_spheres: int,
        tensor_args: Dict[str, Any],
    ) -> None:
        super().__init__(margin, dt, spline_degree, tensor_args)
        self.width = width
        self.height = height
        self.n_spheres = n_spheres
        self.n_dim = 6  # right, base, top

    def enforce_rigid_constraints(self, points: torch.Tensor) -> torch.Tensor:
        right, base, top = points[..., :2], points[..., 2:4], points[..., 4:]
        right_side, top_side = right - base, top - base
        direction_vector = right_side / self.width + top_side / self.height
        sqrt2_inv = 0.5**0.5
        direction_vector = (
            direction_vector
            / torch.linalg.norm(direction_vector, dim=-1, keepdim=True)
            * sqrt2_inv
        )
        direction_vector_sum = direction_vector[..., :1] + direction_vector[..., 1:]
        direction_vector_diff = direction_vector[..., :1] - direction_vector[..., 1:]
        new_right_side = (
            torch.cat([direction_vector_sum, -direction_vector_diff], dim=-1)
            * self.width
        )
        new_top_side = (
            torch.cat([direction_vector_diff, direction_vector_sum], dim=-1)
            * self.height
        )

        new_points = torch.cat(
            [base + new_right_side, base, base + new_top_side], dim=-1
        )

        return new_points

    def get_collision_points(self, points: torch.Tensor) -> torch.Tensor:
        top, base, right = points[..., :2], points[..., 2:4], points[..., 4:]

        n1 = int(self.n_spheres * (self.height / (self.width + self.height)))
        n2 = self.n_spheres - n1 - 1

        l1 = torch.linspace(0, 1, n1 + 1)
        l2 = torch.linspace(0, 1, n2 + 1)

        s1 = top.unsqueeze(-2) * l1.unsqueeze(-1) + base.unsqueeze(-2) * (
            1 - l1
        ).unsqueeze(-1)
        s2 = right.unsqueeze(-2) * l2.unsqueeze(-1) + base.unsqueeze(-2) * (
            1 - l2
        ).unsqueeze(-1)

        all_points = torch.cat([s1, s2[..., 1:, :]], dim=-2)

        return all_points

    def generate_random_points(self, env, n_samples):
        samples = env.random_points((n_samples,))[:, :2]

        x = samples[:, 0]
        y = samples[:, 1]
        theta = torch.rand((n_samples,), **self.tensor_args) * 2 * torch.pi

        right_x = x + self.width * torch.cos(theta)
        right_y = y + self.width * torch.sin(theta)

        top_x = x + self.height * torch.sin(theta)
        top_y = y + self.height * torch.cos(theta)

        points = torch.stack([right_x, right_y, x, y, top_x, top_y], dim=-1)

        return points

    def create_straight_line_trajectory(self, start_pos, goal_pos, n_support_points):
        pass

    def smoothen_trajectory(self, trajectory, n_support_points):
        pass


def get_robots():
    return {
        "Sphere2D": RobotSphere2D,
        "L2D": RobotL2D,
    }
