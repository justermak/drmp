from typing import Any, Dict

import torch

from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.torch_timer import TimerCUDA
from drmp.utils.trajectory_utils import (
    create_straight_line_trajectory,
    smoothen_trajectory
)


class HybridPlanner(ClassicalPlanner):
    def __init__(
        self,
        sample_based_planner: RRTConnect,
        optimization_based_planner: GPMP2,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(
            env=sample_based_planner.env,
            robot=sample_based_planner.robot,
            use_extra_objects=sample_based_planner.use_extra_objects,
            tensor_args=tensor_args,
        )
        self.sample_based_planner = sample_based_planner
        self.optimization_based_planner = optimization_based_planner

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
                trajectories = self.sample_based_planner.optimize(
                    sample_steps=sample_steps,
                    print_freq=print_freq,
                    debug=debug,
                    **kwargs,
                )
                if debug:
                    print(
                        f"Sample-based Planner -- Optimization time: {t_sample_based.elapsed:.3f} sec"
                    )
            with TimerCUDA() as t_smoothen:
                trajectories_smooth = [
                    smoothen_trajectory(
                        traj,
                        n_support_points=self.optimization_based_planner.n_support_points,
                        dt=self.optimization_based_planner.dt,
                    )
                    if traj is not None
                    else create_straight_line_trajectory(
                        start_pos=self.start_pos,
                        goal_pos=self.goal_pos,
                        n_support_points=self.optimization_based_planner.n_support_points,
                        dt=self.optimization_based_planner.dt,
                        tensor_args=self.tensor_args,
                    )
                    for traj in trajectories
                ]
                if debug:
                    print(
                        f"Trajectory Smoothening -- Time: {t_smoothen.elapsed:.3f} sec"
                    )

            init_trajectories = torch.stack(trajectories_smooth)
            torch.cuda.empty_cache()

            with TimerCUDA() as t_opt_based:
                self.optimization_based_planner.reset_trajectories(
                    initial_particle_means=init_trajectories
                )
                trajectories = self.optimization_based_planner.optimize(
                    opt_steps=opt_steps, print_freq=print_freq // 2, debug=debug
                )
                if debug:
                    print(
                        f"Optimization-based Planner -- Optimization time: {t_opt_based.elapsed:.3f} sec"
                    )

            if debug:
                print(
                    f"Hybrid-based Planner -- Optimization time: {t_hybrid.elapsed:.3f} sec"
                )

        return trajectories

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.sample_based_planner.reset(start_pos, goal_pos)
        self.optimization_based_planner.reset(start_pos, goal_pos)
