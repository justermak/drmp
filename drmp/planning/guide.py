from drmp.datasets.dataset import TrajectoryDatasetBSpline, TrajectoryDatasetBase
from drmp.utils.trajectory_utils import get_trajectories_from_bsplines
import torch
from torch import nn

from drmp.config import N_DIM
from drmp.planning.costs import CostComposite


class Guide():
    def __init__(
        self,
        cost: CostComposite,
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        self.cost = cost
        self.n_interpolate = n_interpolate
        self.max_grad_norm = max_grad_norm

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.shape[-1] == N_DIM:
            trajectories = torch.cat([x, torch.zeros_like(x)], dim=-1)
        else:
            trajectories = x.clone()

        with torch.enable_grad():
            trajectories.requires_grad_(True)

            cost = self.cost(
                trajectories=trajectories, n_interpolate=self.n_interpolate
            ).sum()

            grad = torch.autograd.grad(cost, trajectories)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

            if x.shape[-1] == N_DIM:
                grad = grad[..., :N_DIM]
        return -grad


class GuideSlow(nn.Module):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        cost: CostComposite,
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.cost = cost
        self.n_interpolate = n_interpolate
        self.max_grad_norm = max_grad_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == N_DIM:
            trajectories_normalized = torch.cat([x, torch.zeros_like(x)], dim=-1)
        else:
            trajectories_normalized = x.clone()
            
        with torch.enable_grad():
            trajectories_normalized.requires_grad_(True)
            
            trajectories = self.dataset.normalizer.unnormalize(trajectories_normalized) 
            if isinstance(self.dataset, TrajectoryDatasetBSpline):
                trajectories = get_trajectories_from_bsplines(
                    control_points=trajectories,
                    n_support_points=self.dataset.n_support_points,
                    degree=self.dataset.spline_degree,
                )

            cost = self.cost(
                trajectories=trajectories, n_interpolate=self.n_interpolate
            ).sum()

            grad = torch.autograd.grad(cost, trajectories_normalized)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

            if x.shape[-1] == N_DIM:
                grad = grad[..., :N_DIM]
        return -grad
