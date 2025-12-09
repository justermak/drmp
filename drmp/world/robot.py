from typing import Any, Dict

import torch

from drmp.config import N_DIM


class Robot:
    def __init__(
        self,
        margin: float,
        dt: float,
        tensor_args: Dict[str, Any],
    ) -> None:
        self.tensor_args = tensor_args
        self.margin = margin
        self.dt = dt

    def get_position(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :N_DIM]

    def get_velocity(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 2 * N_DIM, (
            "Input tensor must have position and velocity concatenated."
        )
        vel = x[..., N_DIM : 2 * N_DIM]
        return vel

    def invert_trajectories(self, trajectories: torch.Tensor) -> torch.Tensor:
        trajectories_reversed = torch.flip(trajectories, dims=[-2])
        pos_reversed = self.get_position(trajectories_reversed)
        vel_reversed = -self.get_velocity(trajectories_reversed)
        trajectories_inverted = torch.cat([pos_reversed, vel_reversed], dim=-1)
        return trajectories_inverted

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> "Robot":
        """Move tensor_args to the specified device and dtype."""
        if device is not None:
            self.tensor_args["device"] = device
        if dtype is not None:
            self.tensor_args["dtype"] = dtype
        return self
