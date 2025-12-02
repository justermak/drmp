from typing import Dict, Any

import torch

from drmp.config import DEFAULT_TENSOR_ARGS, N_DIMS
from drmp.utils.trajectory_utils import finite_difference_vector


class Robot:   
    def __init__(
        self,
        margin: float = 0.01,
        dt: float = 1.0,
        tensor_args: Dict[str, Any] = DEFAULT_TENSOR_ARGS,
    ) -> None:
        self.margin = margin
        self.tensor_args = tensor_args
        self.dt = dt

    def get_position(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :N_DIMS]

    def get_velocity(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == N_DIMS:
            vel = finite_difference_vector(x, dt=self.dt)
            return vel
        vel = x[..., N_DIMS:2*N_DIMS]
        return vel

    def invert_trajectories(self, trajs: torch.Tensor) -> torch.Tensor:
        trajs_reversed = torch.flip(trajs, dims=[-2])
        pos_reversed = self.get_position(trajs_reversed)
        vel_reversed = -self.get_velocity(trajs_reversed)
        trajs_inverted = torch.cat([pos_reversed, vel_reversed], dim=-1)
        return trajs_inverted