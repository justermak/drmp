from copy import copy
import traceback
from typing import List
import torch
import torch.multiprocessing as mp

from drmp.planning.planners.rrt_connect import RRTConnect


class ParallelSampleBasedPlanner:
    def __init__(
        self,
        planner: RRTConnect,
        n_trajectories: int = 2,
        use_parallel: bool=True,
        max_processes: int =-1,
        start_method: str ="spawn",
    ):
        self.planner = planner
        self.n_trajectories = n_trajectories
        self.use_parallel = use_parallel
        self.max_processes = max_processes
        self.start_method = start_method

        if self.use_parallel:
            self.planners: list[RRTConnect] = []
            for i in range(n_trajectories):
                planner_copy = copy(planner)
                # Assign planner ID for logging
                planner_copy.planner_id = i
                self.planners.append(planner_copy)

    def optimize(self, **kwargs) -> List[torch.Tensor]:
        if self.use_parallel:
            return self._optimize_parallel(**kwargs)
        else:
            return self._optimize_sequential(**kwargs)

    def _optimize_parallel(self, **kwargs) -> List[torch.Tensor]:
        mp.set_start_method(self.start_method, force=True)

        n_processes = (
            min((mp.cpu_count() - 1 if self.max_processes == -1 else self.max_processes), self.n_trajectories)
        )

        with mp.Pool(processes=n_processes) as pool:
            async_results = [
                pool.apply_async(planner.optimize, kwds=kwargs)
                for planner in self.planners
            ]

            trajs = []
            for async_result in async_results:
                try:
                    traj = async_result.get(timeout=kwargs.get("timeout", None))
                    if traj is not None:
                        trajs.append(traj)
                except mp.TimeoutError:
                    print("Trajectory optimization timed out")
                except Exception as e:
                    print(f"Failed to find trajectory: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return trajs

    def _optimize_sequential(self, **kwargs) -> List[torch.Tensor]:
        trajs = []
        for i in range(self.n_trajectories):
            try:
                # Assign planner ID for logging
                self.planner.planner_id = i
                traj = self.planner.optimize(**kwargs)
                trajs.append(traj)
            except Exception as e:
                print(f"Trajectory {i} optimization failed: {e}")
                traceback.print_exc()
        return trajs
