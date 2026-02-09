from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from drmp.utils import (
    get_trajectories_derivative_from_bsplines,
    get_trajectories_from_bsplines,
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


def get_robots():
    return {
        "Sphere2D": RobotSphere2D,
    }
