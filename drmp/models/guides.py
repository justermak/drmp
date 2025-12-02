import torch
from torch import nn

from drmp.datasets.dataset import TrajectoryDataset
from drmp.utils.trajectory_utils import interpolate_trajs


class GuideTrajectories(nn.Module):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        cost,
        do_clip_grad=False,
        max_grad_norm=1.0,
        interpolate_trajectories_for_collision=False,
        num_interpolated_points_for_collision=5,
    ):
        super().__init__()
        self.cost = cost
        self.dataset = dataset

        self.interpolate_trajectories_for_collision = (
            interpolate_trajectories_for_collision
        )
        self.num_interpolated_points_for_collision = (
            num_interpolated_points_for_collision
        )

        self.do_clip_grad = do_clip_grad
        self.max_grad_norm = max_grad_norm

    def forward(self, x_normalized):
        x = x_normalized.clone()
        with torch.enable_grad():
            x.requires_grad_(True)

            x = self.dataset.normalizer.unnormalize(x)

            if self.interpolate_trajectories_for_collision:
                x_interpolated = interpolate_trajs(
                    x,
                    n_interp=self.num_interpolated_points_for_collision,
                )
            else:
                x_interpolated = x

            costs, weights = self.cost(
                x, x_interpolated=x_interpolated, return_invidual_costs_and_weights=True
            )
            grad = 0
            for cost, weight in zip(costs, weights):
                if torch.is_tensor(cost):
                    grad_cost = torch.autograd.grad(
                        [cost.sum()], [x], retain_graph=True
                    )[0]

                    grad_cost_clipped = self.clip_grad(grad_cost)

                    grad_cost_clipped[..., 0, :] = 0.0
                    grad_cost_clipped[..., -1, :] = 0.0

                    grad_cost_clipped_weighted = weight * grad_cost_clipped
                    grad += grad_cost_clipped_weighted

        grad = -1.0 * grad
        return grad

    def clip_grad(self, grad, eps=1e-6):
        if self.do_clip_grad:
            grad_norm = torch.linalg.norm(grad + eps, dim=-1, keepdims=True)
            scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
            grad = scale_ratio * grad
        return grad
