import torch
from torch import nn

from drmp.config import N_DIM
from drmp.datasets.dataset import TrajectoryDatasetBase, TrajectoryDatasetBSpline
from drmp.planning.costs.cost_functions import CostComposite
from drmp.utils.trajectory_utils import get_trajectories_from_bsplines


class GuideTrajectories(nn.Module):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        cost: CostComposite,
        do_clip_grad: bool,
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        super().__init__()
        self.cost = cost
        self.dataset = dataset

        self.n_interpolate = n_interpolate
        self.do_clip_grad = do_clip_grad
        self.max_grad_norm = max_grad_norm

    def forward(self, trajectories_normalized: torch.Tensor, **kwargs) -> torch.Tensor:
        t_normalized = trajectories_normalized.clone()
        if trajectories_normalized.shape[-1] == N_DIM:
            t_normalized = torch.cat(
                [t_normalized, torch.zeros_like(t_normalized)], dim=-1
            )

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
            if self.do_clip_grad:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

            grad = -1.0 * grad

            if trajectories_normalized.shape[-1] == N_DIM:
                grad = grad[..., :N_DIM]
        return grad
