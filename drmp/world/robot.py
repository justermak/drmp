from typing import Any, Dict

from abc import ABC, abstractmethod

import torch

class RobotBase(ABC):
    def __init__(
        self,
        margin: float,
        dt: float,
        tensor_args: Dict[str, Any],
    ) -> None:
        self.tensor_args = tensor_args
        self.margin = margin
        self.dt = dt
        self.n_dim: int = None
        
    @abstractmethod
    def get_position(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_velocity(self, x: torch.Tensor) -> torch.Tensor:
        pass    
    
    @abstractmethod
    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        pass
    
    
class RobotSphere2D(RobotBase):
    def __init__(
        self,
        margin: float,
        dt: float,
        tensor_args: Dict[str, Any],
    ) -> None:
        super().__init__(margin, dt, tensor_args)
        self.n_dim = 2

    def get_position(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.n_dim]

    def get_velocity(self, x: torch.Tensor, compute: bool = False) -> torch.Tensor:
        if compute:
            pos = self.get_position(x)
            vel = (pos[..., 1:, :] - pos[..., :-1, :]) / self.dt
            vel = torch.cat([torch.zeros_like(pos[..., :1, :]), vel], dim=-2)
            return vel
        assert x.shape[-1] >= 2 * self.n_dim, (
            "Input tensor must have position and velocity concatenated."
        )
        vel = x[..., self.n_dim : 2 * self.n_dim]
        return vel

    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.shape[-1] == self.n_dim:
            trajectories_inverted = torch.flip(trajectories, dims=[-2])
            return trajectories_inverted
        
        trajectories_reversed = torch.flip(trajectories, dims=[-2])
        pos_reversed = self.get_position(trajectories_reversed)
        vel_reversed = -self.get_velocity(trajectories_reversed)
        trajectories_inverted = torch.cat([pos_reversed, vel_reversed], dim=-1)
        return trajectories_inverted


def get_robots():
    return {
        "Sphere2D": RobotSphere2D,
    }