from typing import List

import torch

from drmp.datasets.dataset import TrajectoryDatasetBase, TrajectoryDatasetBSpline
from drmp.planning.costs import Cost


class GuideBase:
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        costs: List[Cost],
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        self.dataset = dataset
        self.costs = costs
        self.n_interpolate = n_interpolate
        self.max_grad_norm = max_grad_norm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Guide(GuideBase):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        costs: List[Cost],
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        super().__init__(
            dataset=dataset,
            costs=costs,
            max_grad_norm=max_grad_norm,
            n_interpolate=n_interpolate,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        trajectories_normalized = x.clone()

        with torch.enable_grad():
            trajectories_normalized.requires_grad_(True)

            trajectories = self.dataset.normalizer.unnormalize(trajectories_normalized)
            if isinstance(self.dataset, TrajectoryDatasetBSpline):
                trajectories_pos = self.dataset.robot.get_position_interpolated(
                    trajectories=trajectories,
                    n_support_points=self.dataset.n_support_points,
                )
            else:
                trajectories_pos = self.dataset.robot.get_position(
                    trajectories=trajectories,
                )
            cost = sum(
                cost(
                    trajectories=trajectories_pos,
                    n_interpolate=self.n_interpolate,
                ).sum()
                for cost in self.costs
            )

            grad = torch.autograd.grad(cost, trajectories_normalized)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

        return -grad


class GuideDense(GuideBase):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        costs: List[Cost],
        max_grad_norm: float,
        n_interpolate: int,
    ) -> None:
        super().__init__(
            dataset=dataset,
            costs=costs,
            max_grad_norm=max_grad_norm,
            n_interpolate=n_interpolate,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        trajectories = x.clone()

        with torch.enable_grad():
            trajectories.requires_grad_(True)
            trajectories_pos = self.dataset.robot.get_position(
                trajectories=trajectories,
            )

            cost = sum(
                cost(
                    trajectories=trajectories_pos,
                    n_interpolate=self.n_interpolate,
                ).sum()
                for cost in self.costs
            )

            grad = torch.autograd.grad(cost, trajectories)[0]
            if self.max_grad_norm is not None:
                grad_norm = torch.linalg.norm(grad + 1e-8, dim=-1, keepdims=True)
                scale_ratio = torch.clip(grad_norm, 0.0, self.max_grad_norm) / grad_norm
                grad = scale_ratio * grad

        return -grad
