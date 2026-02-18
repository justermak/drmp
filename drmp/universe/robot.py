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
    def get_position(
        self,
        trajectories: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_position_interpolated(
        self,
        control_points: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def fit_bsplines_to_position(
        self,
        trajectories: torch.Tensor,
        n_control_points: int,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_velocity(
        self,
        trajectories: torch.Tensor,
        mode: str = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_acceleration(
        self,
        trajectories: torch.Tensor,
        mode: str = None,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_collision_mask(
        self,
        env: EnvBase,
        points: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_cost(
        self,
        env: EnvBase,
        trajectories: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def random_collision_free_points(
        self,
        env: EnvBase,
        n_samples: int,
        use_extra_objects: bool = False,
        batch_size=100000,
        max_tries=1000,
    ) -> Tuple[torch.Tensor, bool]:
        pass

    @abstractmethod
    def random_collision_free_start_goal(
        self,
        env: EnvBase,
        n_samples: int,
        threshold_start_goal_pos: float,
        use_extra_objects: bool = False,
        batch_size: int = 100000,
        max_tries: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        pass

    @abstractmethod
    def get_trajectories_collision_and_free(
        self,
        env: EnvBase,
        trajectories: torch.Tensor,
        n_interpolate: int = 5,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @abstractmethod
    def gradient_step(
        self, trajectories: torch.Tensor, grad: torch.Tensor
    ) -> torch.Tensor:
        pass


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
        return trajectories_pos

    def fit_bsplines_to_position(
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
    ) -> torch.Tensor:
        if mode == "forward":
            trajectories_pos = self.get_position(trajectories)
            trajectories_vel = torch.diff(trajectories_pos, dim=-2) / self.dt
            trajectories_vel = torch.cat(
                [trajectories_vel, torch.zeros_like(trajectories_vel[:, :1, :])], dim=-2
            )
            trajectories_vel[:, 0, :] = 0.0
            return trajectories_vel
        elif mode == "central":
            trajectories_pos = self.get_position(trajectories)
            trajectories_vel = (
                trajectories_pos[:, 2:, :] - trajectories_pos[:, :-2, :]
            ) / (2 * self.dt)
            trajectories_vel = torch.cat(
                [
                    torch.zeros_like(trajectories_vel[:, :1, :]),
                    trajectories_vel,
                    torch.zeros_like(trajectories_vel[:, :1, :]),
                ],
                dim=-2,
            )
            return trajectories_vel
        elif mode == "avg":
            trajectories_pos = self.get_position(trajectories)
            displacement = trajectories_pos[:, -1, :] - trajectories_pos[:, 0, :]
            avg_vel = displacement / (self.dt * (trajectories_pos.shape[1] - 2))
            trajectories_vel = torch.cat(
                [
                    torch.zeros_like(avg_vel).unsqueeze(1),
                    avg_vel.unsqueeze(1).repeat(1, trajectories_pos.shape[1] - 2, 1),
                    torch.zeros_like(avg_vel).unsqueeze(1),
                ],
                dim=1,
            )
            return trajectories_vel
        if trajectories.shape[-1] >= 2 * self.n_dim:
            trajectories_vel = trajectories[..., self.n_dim : 2 * self.n_dim]
        else:
            trajectories_vel = torch.zeros_like(trajectories[..., : self.n_dim])
        return trajectories_vel

    def get_acceleration(
        self,
        trajectories: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "forward":
            trajectories_vel = self.get_velocity(trajectories, mode="forward")
            trajectories_acc = self.get_velocity(trajectories_vel, mode="forward")
            return trajectories_acc
        elif mode == "central":
            trajectories_vel = self.get_velocity(trajectories, mode="central")
            trajectories_acc = self.get_velocity(trajectories_vel, mode="central")
            return trajectories_acc

    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.shape[-1] == self.n_dim:
            trajectories_inverted = torch.flip(trajectories, dims=[-2])
            return trajectories_inverted

        assert trajectories.shape[-1] >= 2 * self.n_dim, (
            "Input tensor must have position and velocity concatenated."
        )
        trajectories_reversed = torch.flip(trajectories, dims=[-2])
        trajectories_pos_reversed = self.get_position(trajectories_reversed)
        trajectories_vel_reversed = -self.get_velocity(trajectories_reversed)
        trajectories_inverted = torch.cat(
            [trajectories_pos_reversed, trajectories_vel_reversed], dim=-1
        )
        return trajectories_inverted

    def get_collision_mask(
        self,
        env: EnvBase,
        points: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        points = self.get_position(points)
        collision_mask_fixed = torch.zeros(
            points.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        collision_mask_extra = torch.zeros(
            points.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(points)
            collision_mask_fixed = (sdf_fixed < self.margin).any(dim=-1)
        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(points)
            collision_mask_extra = (sdf_extra < self.margin).any(dim=-1)

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
        total_cost = torch.zeros(trajectories_pos.shape[:-1], **self.tensor_args)
        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(
                trajectories_pos
            )
            cost_fixed = torch.relu(self.margin - sdf_fixed).sum(dim=-1)
            total_cost += cost_fixed
        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(
                trajectories_pos
            )
            cost_extra = torch.relu(self.margin - sdf_extra).sum(dim=-1)
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
            points = env.random_points((batch_size,))
            collision_mask = self.get_collision_mask(
                env=env, points=points, on_extra=use_extra_objects
            ).squeeze()
            n = torch.sum(~collision_mask).item()
            n = min(n, n_samples - cur)
            samples[cur : cur + n] = points[~collision_mask][:n]
            cur += n
            if cur >= n_samples:
                break

        return samples.squeeze(), cur >= n_samples

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
            start_state_pos, goal_state_pos = points[:n_samples], points[n_samples:]
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

    def gradient_step(
        self, trajectories: torch.Tensor, grad: torch.Tensor
    ) -> torch.Tensor:
        return trajectories + grad


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
        self.n_dim = 6

        self.points = torch.tensor([[width, 0], [0, 0], [0, height]], **tensor_args)

    def _enforce_rigid_constraints(self, points: torch.Tensor) -> torch.Tensor:
        top, base, right = points[..., :2], points[..., 2:4], points[..., 4:]
        top_side, right_side = top - base, right - base
        direction_vector = top_side / self.height + right_side / self.width
        sqrt2_inv = 0.5**0.5
        direction_vector = (
            direction_vector
            / torch.linalg.norm(direction_vector, dim=-1, keepdim=True)
            * sqrt2_inv
        )
        direction_vector_sum = direction_vector[..., :1] + direction_vector[..., 1:]
        direction_vector_diff = direction_vector[..., :1] - direction_vector[..., 1:]
        new_top_side = (
            torch.cat([direction_vector_diff, direction_vector_sum], dim=-1)
            * self.height
        )
        new_right_side = (
            torch.cat([direction_vector_sum, -direction_vector_diff], dim=-1)
            * self.width
        )

        new_points = torch.cat(
            [base + new_top_side, base, base + new_right_side], dim=-1
        )

        return new_points

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
        trajectories_pos = self._enforce_rigid_constraints(trajectories_pos)
        return trajectories_pos

    def fit_bsplines_to_position(
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
    ) -> torch.Tensor:
        if mode == "forward":
            trajectories_pos = self.get_position(trajectories)
            trajectories_vel = torch.diff(trajectories_pos, dim=-2) / self.dt
            trajectories_vel = torch.cat(
                [trajectories_vel, torch.zeros_like(trajectories_vel[:, :1, :])], dim=-2
            )
            trajectories_vel[:, 0, :] = 0.0
            return trajectories_vel
        elif mode == "central":
            trajectories_pos = self.get_position(trajectories)
            trajectories_vel = (
                trajectories_pos[:, 2:, :] - trajectories_pos[:, :-2, :]
            ) / (2 * self.dt)
            trajectories_vel = torch.cat(
                [
                    torch.zeros_like(trajectories_vel[:, :1, :]),
                    trajectories_vel,
                    torch.zeros_like(trajectories_vel[:, :1, :]),
                ],
                dim=-2,
            )
            return trajectories_vel
        elif mode == "avg":
            trajectories_pos = self.get_position(trajectories)
            displacement = trajectories_pos[:, -1, :] - trajectories_pos[:, 0, :]
            avg_vel = displacement / (self.dt * (trajectories_pos.shape[1] - 2))
            trajectories_vel = torch.cat(
                [
                    torch.zeros_like(avg_vel).unsqueeze(1),
                    avg_vel.unsqueeze(1).repeat(1, trajectories_pos.shape[1] - 2, 1),
                    torch.zeros_like(avg_vel).unsqueeze(1),
                ],
                dim=1,
            )
            return trajectories_vel
        if trajectories.shape[-1] >= 2 * self.n_dim:
            trajectories_vel = trajectories[..., self.n_dim : 2 * self.n_dim]
        else:
            trajectories_vel = torch.zeros_like(trajectories[..., : self.n_dim])
        return trajectories_vel

    def get_acceleration(
        self,
        trajectories: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "forward":
            trajectories_vel = self.get_velocity(trajectories, mode="forward")
            trajectories_acc = self.get_velocity(trajectories_vel, mode="forward")
            return trajectories_acc
        elif mode == "central":
            trajectories_vel = self.get_velocity(trajectories, mode="central")
            trajectories_acc = self.get_velocity(trajectories_vel, mode="central")
            return trajectories_acc

    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.shape[-1] == self.n_dim:
            trajectories_inverted = torch.flip(trajectories, dims=[-2])
            return trajectories_inverted

        assert trajectories.shape[-1] >= 2 * self.n_dim, (
            "Input tensor must have position and velocity concatenated."
        )
        trajectories_reversed = torch.flip(trajectories, dims=[-2])
        trajectories_pos_reversed = self.get_position(trajectories_reversed)
        trajectories_vel_reversed = -self.get_velocity(trajectories_reversed)
        trajectories_inverted = torch.cat(
            [trajectories_pos_reversed, trajectories_vel_reversed], dim=-1
        )
        return trajectories_inverted

    def _get_collision_points(self, points: torch.Tensor) -> torch.Tensor:
        p1 = points[..., 0:2]
        p2 = points[..., 2:4]
        p3 = points[..., 4:6]

        # Distribute spheres proportional to length
        total_len = self.width + self.height
        n1 = int(self.n_spheres * (self.width / total_len))
        n2 = self.n_spheres - n1

        # Ensure at least 1 sphere per segment if possible, unless length is 0?
        if n1 == 0 and self.width > 0:
            n1 = 1
            n2 -= 1
        if n2 == 0 and self.height > 0:
            n2 = 1
            n1 -= 1

        if n1 > 1:
            w1 = torch.linspace(0, 1, n1, device=points.device, dtype=points.dtype)
        else:
            w1 = torch.tensor([0.5], device=points.device, dtype=points.dtype)

        if n2 > 1:
            w2 = torch.linspace(0, 1, n2, device=points.device, dtype=points.dtype)
        else:
            w2 = torch.tensor([0.5], device=points.device, dtype=points.dtype)

        seg1_points = p1.unsqueeze(-2) * (1 - w1.unsqueeze(-1)) + p2.unsqueeze(
            -2
        ) * w1.unsqueeze(-1)
        seg2_points = p2.unsqueeze(-2) * (1 - w2.unsqueeze(-1)) + p3.unsqueeze(
            -2
        ) * w2.unsqueeze(-1)

        all_points = torch.cat([seg1_points, seg2_points], dim=-2)
        return all_points

    def get_collision_mask(
        self,
        env: EnvBase,
        points: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        points_pos = self.get_position(points)
        collision_points = self._get_collision_points(points_pos)
        # collision_points shape: (..., n_spheres, 2)

        collision_mask_fixed = torch.zeros(
            points_pos.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        collision_mask_extra = torch.zeros(
            points_pos.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )

        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(
                collision_points
            )
            # sdf_fixed shape: (..., n_spheres)
            # Check if ANY sphere is in collision
            collision_mask_fixed = (sdf_fixed < self.margin).any(dim=-1)

        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(
                collision_points
            )
            collision_mask_extra = (sdf_extra < self.margin).any(dim=-1)

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
        collision_points = self._get_collision_points(trajectories_pos)

        total_cost = torch.zeros(trajectories_pos.shape[:-1], **self.tensor_args)

        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(
                collision_points
            )
            # Cost is sum of penetrations across all spheres
            cost_fixed = torch.relu(self.margin - sdf_fixed).sum(dim=-1)
            total_cost += cost_fixed

        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(
                collision_points
            )
            cost_extra = torch.relu(self.margin - sdf_extra).sum(dim=-1)
            total_cost += cost_extra

        return total_cost

    def _state_from_pose(self, pos_x, pos_y, theta):
        # Generate 3 points from rigid pose (x, y, theta)
        # Assume (x, y) is the corner (p2)
        # p1 is along theta
        # p3 is along theta + 90

        # Dimensions are self.width and self.height
        # Let width be p1-p2 length, height be p2-p3 length

        p2_x = pos_x
        p2_y = pos_y

        p1_x = p2_x + self.width * torch.cos(theta)
        p1_y = p2_y + self.width * torch.sin(theta)

        p3_x = p2_x + self.height * torch.cos(theta + torch.pi / 2)
        p3_y = p2_y + self.height * torch.sin(theta + torch.pi / 2)

        # Stack into (..., 6)
        return torch.stack([p1_x, p1_y, p2_x, p2_y, p3_x, p3_y], dim=-1)

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
            # Generate random poses: x, y, theta
            # env.random_points usually returns (batch, 2) for 2D envs
            # We need to add random theta

            random_pos = env.random_points((batch_size,))  # (batch, 2)
            random_theta = torch.rand((batch_size,), **self.tensor_args) * 2 * torch.pi

            points_6d = self._state_from_pose(
                random_pos[:, 0], random_pos[:, 1], random_theta
            )

            collision_mask = self.get_collision_mask(
                env=env, points=points_6d, on_extra=use_extra_objects
            ).squeeze()

            n = torch.sum(~collision_mask).item()
            n = min(n, n_samples - cur)

            valid_samples = points_6d[~collision_mask]

            # Additional logic to handle if valid_samples is smaller than expected due to filtering
            if n > 0:
                samples[cur : cur + n] = valid_samples[:n]
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
            if not success and cur == 0:
                # If we can't even get enough valid samples in one go, we might struggle
                # But we loop max_tries times, so let's continue.
                pass

            # Note: random_collision_free_points returns (n_samp_req, 6)
            # If it failed to fill, it returns what it has up to n_samp_req
            # BUT the implementation above initializes zeros of size n_samples
            # So if success is False, it might contain zeros at the end.

            # Let's just use what we got
            # We need pairs.

            # Actually random_collision_free_points returns (n_samples, 6) and a bool.
            # If bool is False, it might be partially filled.

            # We split points into candidates
            n_candidates = points.shape[0] // 2
            start_candidates = points[:n_candidates]
            goal_candidates = points[n_candidates : 2 * n_candidates]

            # Check distance. For 6D, use Frobenius norm? Or max displacement?
            # User said "threshold_start_goal_pos".
            # For robot, maybe distance of center of mass (p2)?
            # Let's use checking distance between the corners (p2).

            p2_start = start_candidates[:, 2:4]
            p2_goal = goal_candidates[:, 2:4]

            dist = torch.linalg.norm(p2_start - p2_goal, dim=-1)
            threshold_mask = dist > threshold_start_goal_pos

            n = torch.sum(threshold_mask).item()
            n = min(n, n_samples - cur)

            if n > 0:
                samples_start[cur : cur + n] = start_candidates[threshold_mask][:n]
                samples_goal[cur : cur + n] = goal_candidates[threshold_mask][:n]
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
    
    def create_straight_line_trajectory(self, start_pos, goal_pos, n_support_points):
        pass
    
    def smoothen_trajectory(self, trajectory, n_support_points):
        pass

    def gradient_step(
        self, trajectories: torch.Tensor, grad: torch.Tensor
    ) -> torch.Tensor:
        new_trajectories = trajectories + grad
        constrained_trajectories = self._enforce_rigid_constraints(new_trajectories)
        return constrained_trajectories


def get_robots():
    return {
        "Sphere2D": RobotSphere2D,
        "L2D": RobotL2D,
    }
