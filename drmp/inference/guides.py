import torch
from torch import nn

from drmp.config import N_DIM
from drmp.datasets.dataset import TrajectoryDatasetBase, TrajectoryDatasetBSpline
from drmp.planning.costs.cost_functions import CostComposite
from drmp.utils.trajectory_utils import get_trajectories_from_bsplines


class Guide(nn.Module):
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.shape[-1] == N_DIM:
            t_normalized = torch.cat([x, torch.zeros_like(x)], dim=-1)
        else:
            t_normalized = x.clone()

        with torch.enable_grad():
            t_normalized.requires_grad_(True)

            t = self.dataset.normalizer.unnormalize(t_normalized)
            if isinstance(self.dataset, TrajectoryDatasetBSpline):
                trajectories_dense = get_trajectories_from_bsplines(
                    t,
                    n_support_points=self.dataset.n_support_points,
                    degree=self.dataset.spline_degree,
                )
            else:
                trajectories_dense = t

            cost = self.cost(
                trajectories=trajectories_dense, n_interpolate=self.n_interpolate
            ).sum()

            grad = torch.autograd.grad(cost, t_normalized)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

            grad = -1.0 * grad

            if x.shape[-1] == N_DIM:
                grad = grad[..., :N_DIM]
        return grad
