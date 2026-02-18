from typing import Any, Dict, List, Optional

import torch

from drmp.dataset.transform import NormalizerBase
from drmp.planning.costs import Cost
from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.torch_timer import TimerCUDA
from drmp.universe.environments import EnvBase
from drmp.universe.robot import RobotBase


class GradientOptimization(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
        normalizer: NormalizerBase,
        n_support_points: int,
        n_control_points: Optional[int],
        costs: List[Cost],
        max_grad_norm: float,
        n_interpolate: int,
        tensor_args: Dict[str, Any],
        use_extra_objects: bool = False,
    ):
        super().__init__(env, robot, use_extra_objects, tensor_args)
        self.normalizer = normalizer
        self.n_support_points = n_support_points
        self.n_control_points = n_control_points
        self.costs = costs
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        self.start_pos = start_pos
        self.goal_pos = goal_pos

    def compute_gradient(
        self,
        x: torch.Tensor,
        return_cost: bool = False,
    ) -> torch.Tensor:
        trajectories_normalized = x.clone()

        with torch.enable_grad():
            trajectories_normalized.requires_grad_(True)
            trajectories_unnormalized = self.normalizer.unnormalize(
                trajectories_normalized
            )

            if self.n_control_points is not None:
                trajectories_pos = self.robot.get_position_interpolated(
                    control_points=trajectories_unnormalized,
                    n_support_points=self.n_support_points,
                )
            else:
                trajectories_pos = self.robot.get_position(
                    trajectories=trajectories_unnormalized,
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

            n_fixed = 2 if self.n_control_points is not None else 1
            grad[..., :n_fixed, :] = 0.0
            grad[..., -n_fixed:, :] = 0.0

        if return_cost:
            return grad, cost
        return grad

    def __call__(self, trajectories: torch.Tensor) -> torch.Tensor:
        return -self.compute_gradient(
            x=trajectories,
            return_cost=False,
        )

    def optimize(
        self,
        trajectories: torch.Tensor,
        n_optimization_steps: Optional[int],
        print_freq: Optional[int] = None,
        debug: bool = False,
    ) -> torch.Tensor:
        x = trajectories.clone()
        if self.n_control_points is not None and x.shape[-2] == self.n_support_points:
            x = self.robot.get_position_interpolated(
                control_points=x,
                n_support_points=self.n_support_points,
            )
        else:
            x = self.normalizer.normalize(x)
        cost = None
        with TimerCUDA() as t_opt:
            for i in range(n_optimization_steps):
                if debug and print_freq and i % print_freq == 0:
                    self.print_info(i + 1, t_opt.elapsed, cost)

                grad, cost = self.compute_gradient(x=x, return_cost=True)

                x = self.robot.gradient_step(x, -grad)

        return x

    def print_info(
        self, step: int, elapsed_time: float, cost: torch.Tensor = None
    ) -> None:
        cost_val = cost.item() if cost is not None else float("nan")
        print(f"Step {step} | Time: {elapsed_time:.3f} sec | Cost: {cost_val:.3e}")
