from typing import Any, Dict

import torch

from drmp.config import DEFAULT_TENSOR_ARGS
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.parallel_sample_based_planner import (
    ParallelSampleBasedPlanner,
)
from drmp.utils.torch_timer import TimerCUDA
from drmp.utils.trajectory_utils import smoothen_trajectory


class HybridPlanner:
    """
    Runs a sampled-based planner to get an initial trajectory in position, followed by an optimization-based planner
    to optimize for positions and velocities.
    """

    def __init__(
        self,
        sample_based_planner: ParallelSampleBasedPlanner,
        opt_based_planner: GPMP2,
        tensor_args: Dict[str, Any] = DEFAULT_TENSOR_ARGS,
    ):
        self.tensor_args = tensor_args
        self.sample_based_planner = sample_based_planner
        self.opt_based_planner = opt_based_planner

    def optimize(
        self,
        sample_steps: int,
        opt_steps: int,
        print_freq: int = 200,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        with TimerCUDA() as t_hybrid:
            with TimerCUDA() as t_sample_based:
                trajs = self.sample_based_planner.optimize(
                    sample_steps=sample_steps,
                    print_freq=print_freq,
                    debug=debug,
                    **kwargs,
                )
            if debug:
                print(
                    f"Sample-based Planner -- Optimization time: {t_sample_based.elapsed:.3f} sec"
                )

            trajs_smooth = [
                smoothen_trajectory(
                    traj,
                    n_support_points=self.opt_based_planner.n_support_points,
                    dt=self.opt_based_planner.dt,
                    tensor_args=self.tensor_args,
                )
                for traj in trajs
            ]

            init_trajs = torch.stack(trajs_smooth)
            torch.cuda.empty_cache()

            with TimerCUDA() as t_opt_based:
                self.opt_based_planner.reset_trajectories(
                    initial_particle_means=init_trajs
                )
                trajs = self.opt_based_planner.optimize(
                    opt_steps=opt_steps, print_freq=print_freq, debug=debug
                )
            if debug:
                print(
                    f"Optimization-based Planner -- Optimization time: {t_opt_based.elapsed:.3f} sec"
                )

        if debug:
            print(
                f"Hybrid-based Planner -- Optimization time: {t_hybrid.elapsed:.3f} sec"
            )

        return trajs

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        self.sample_based_planner.reset(start_pos, goal_pos)
        self.opt_based_planner.reset(start_pos, goal_pos.unsqueeze(0))

    def shutdown(self) -> None:
        self.sample_based_planner.shutdown()
