import torch
from torch import nn

from drmp.datasets.dataset import TrajectoryDataset
from drmp.planning.costs.cost_functions import CostComposite


class GuideTrajectories(nn.Module):
    def __init__(
        self,
        dataset: TrajectoryDataset,
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

    def forward(self, x_normalized: torch.Tensor) -> torch.Tensor:
        x = x_normalized.clone()
        with torch.enable_grad():
            x.requires_grad_(True)

            x = self.dataset.normalizer.unnormalize(x)

            cost = self.cost(trajectories=x, n_interpolate=self.n_interpolate).sum()

            grad = torch.autograd.grad(cost, x)[0]
            if self.do_clip_grad:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

            grad = -1.0 * grad
        return grad
