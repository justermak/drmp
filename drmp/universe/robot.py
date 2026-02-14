from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch

from drmp.utils import (
    get_trajectories_derivative_from_bsplines,
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
        trajectories: torch.Tensor,
        n_support_points: int,
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
    def get_velocity_interpolated(
        self,
        trajectories: torch.Tensor,
        n_support_points: int,
        trajectories_pos: torch.Tensor = None,
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
        env,
        qs: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_cost(
        self,
        env,
        trajectories: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def random_collision_free_q(
        self,
        env,
        n_samples: int,
        use_extra_objects: bool = False,
        batch_size=100000,
        max_tries=1000,
    ) -> Tuple[torch.Tensor, bool]:
        pass

    @abstractmethod
    def random_collision_free_start_goal(
        self,
        env,
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
        env,
        trajectories: torch.Tensor,
        n_interpolate: int = 5,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        trajectories: torch.Tensor,
        n_support_points: int,
    ) -> torch.Tensor:
        trajectories_pos = get_trajectories_from_bsplines(
            control_points=trajectories[..., : self.n_dim],
            n_support_points=n_support_points,
            degree=self.spline_degree,
        )
        return trajectories_pos

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

    def get_velocity_interpolated(
        self,
        trajectories: torch.Tensor,
        n_support_points: int,
        trajectories_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        trajectories_pos = (
            trajectories_pos
            if trajectories_pos is not None
            else self.get_position_interpolated(
                trajectories, n_support_points=n_support_points
            )
        )
        trajectories_vel = get_trajectories_derivative_from_bsplines(
            control_points=trajectories_pos,
            n_support_points=n_support_points,
            degree=self.spline_degree,
        ) / (self.dt * (n_support_points - 1))
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
        env,
        qs: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        """Check collision for given configurations.
        
        Args:
            env: Environment object with grid_map_sdf_fixed and grid_map_sdf_extra
            qs: Configuration tensor
            on_fixed: Whether to check collision with fixed obstacles
            on_extra: Whether to check collision with extra obstacles
            
        Returns:
            Boolean mask indicating collision
        """
        qs = self.get_position(qs)
        collision_mask_fixed = torch.zeros(
            qs.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        collision_mask_extra = torch.zeros(
            qs.shape[:-1], dtype=torch.bool, device=self.tensor_args["device"]
        )
        if on_fixed:
            sdf_fixed = env.grid_map_sdf_fixed.compute_approx_signed_distance(qs)
            collision_mask_fixed = (sdf_fixed < self.margin).any(dim=-1)
        if on_extra:
            sdf_extra = env.grid_map_sdf_extra.compute_approx_signed_distance(qs)
            collision_mask_extra = (sdf_extra < self.margin).any(dim=-1)

        collision_mask = collision_mask_fixed | collision_mask_extra
        return collision_mask

    def compute_cost(
        self,
        env,
        trajectories: torch.Tensor,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> torch.Tensor:
        """Compute collision cost for given trajectories.
        
        Args:
            env: Environment object with grid_map_sdf_fixed and grid_map_sdf_extra
            trajectories: Trajectory tensor
            on_fixed: Whether to compute cost with fixed obstacles
            on_extra: Whether to compute cost with extra obstacles
            
        Returns:
            Cost tensor
        """
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

    def random_collision_free_q(
        self,
        env,
        n_samples: int,
        use_extra_objects: bool = False,
        batch_size=100000,
        max_tries=1000,
    ) -> Tuple[torch.Tensor, bool]:
        """Generate random collision-free configurations.
        
        Args:
            env: Environment object
            n_samples: Number of collision-free samples to generate
            use_extra_objects: Whether to check collision with extra objects
            batch_size: Batch size for sampling
            max_tries: Maximum number of tries
            
        Returns:
            Tuple of (samples, success flag)
        """
        samples = torch.zeros((n_samples, self.n_dim), **self.tensor_args)
        cur = 0
        for i in range(max_tries):
            qs = env.random_q((batch_size,))
            collision_mask = self.get_collision_mask(
                env=env, qs=qs, on_extra=use_extra_objects
            ).squeeze()
            n = torch.sum(~collision_mask).item()
            n = min(n, n_samples - cur)
            samples[cur : cur + n] = qs[~collision_mask][:n]
            cur += n
            if cur >= n_samples:
                break

        return samples.squeeze(), cur >= n_samples

    def random_collision_free_start_goal(
        self,
        env,
        n_samples: int,
        threshold_start_goal_pos: float,
        use_extra_objects: bool = False,
        batch_size: int = 100000,
        max_tries: int = 1000,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Generate random collision-free start-goal pairs.
        
        Args:
            env: Environment object
            n_samples: Number of start-goal pairs to generate
            threshold_start_goal_pos: Minimum distance between start and goal
            use_extra_objects: Whether to check collision with extra objects
            batch_size: Batch size for sampling
            max_tries: Maximum number of tries
            
        Returns:
            Tuple of (start_samples, goal_samples, success flag)
        """
        samples_start = torch.zeros((n_samples, self.n_dim), **self.tensor_args)
        samples_goal = torch.zeros((n_samples, self.n_dim), **self.tensor_args)
        cur = 0
        for _ in range(max_tries):
            qs, success = self.random_collision_free_q(
                env=env,
                n_samples=n_samples * 2,
                use_extra_objects=use_extra_objects,
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
        env,
        trajectories: torch.Tensor,
        n_interpolate: int = 5,
        on_fixed: bool = True,
        on_extra: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separate trajectories into collision and collision-free.
        
        Args:
            env: Environment object
            trajectories: Trajectory tensor
            n_interpolate: Number of interpolation points
            on_fixed: Whether to check collision with fixed obstacles
            on_extra: Whether to check collision with extra obstacles
            
        Returns:
            Tuple of (collision_trajectories, free_trajectories, points_collision_mask)
        """
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


def get_robots():
    return {
        "Sphere2D": RobotSphere2D,
    }
