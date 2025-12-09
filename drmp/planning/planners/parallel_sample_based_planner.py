import traceback
from copy import deepcopy
from time import sleep
from typing import List

import torch
import torch.multiprocessing as mp

from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.torch_utils import fix_random_seed

_worker_planner: RRTConnect = None


def _worker_init(
    planner_template: RRTConnect, device: torch.device, id_counter, lock, seed: int
) -> bool:
    global _worker_planner
    try:
        with lock:
            worker_id = id_counter.value
            fix_random_seed(seed + worker_id)

            _worker_planner = deepcopy(planner_template)
            _worker_planner.planner_id = worker_id
            _worker_planner.to(device=device)
            _worker_planner._initialize_samples()
            print(f"Worker {worker_id} initialized on device {device}")
            id_counter.value += 1
    except Exception as e:
        print(f"Worker initialization failed: {e}")
        traceback.print_exc()
        return False
    finally:
        torch.cuda.empty_cache()
    return True


def _worker_process_command(command: dict):
    global _worker_planner

    cmd_type = command.get("type")

    if cmd_type == "reset":
        start_pos = command["start_pos"]
        goal_pos = command["goal_pos"]
        _worker_planner.reset(start_pos, goal_pos)
        return {"status": "reset_done"}

    elif cmd_type == "optimize":
        try:
            kwargs = command["kwargs"]
            kwargs["traj_id"] = command["traj_id"]

            result = _worker_planner.optimize(**kwargs)

            if result is not None:
                return {"status": "success", "result": result.detach().cpu()}
            return {"status": "failed", "result": None}
        except Exception as e:
            print(f"Trajectory {command['traj_id']} optimization failed: {e}")
            traceback.print_exc()
            return {"status": "error", "result": None}

    return {"status": "unknown_command"}


class ParallelSampleBasedPlanner:
    def __init__(
        self,
        planner: RRTConnect,
        n_trajectories: int,
        use_parallel: bool,
        max_processes: int,
        seed: int,
    ):
        self.planner = planner
        self.n_trajectories = n_trajectories
        self.use_parallel = use_parallel
        self.max_processes = max_processes
        self.seed = seed
        if self.use_parallel:
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            self._n_workers = min(
                mp.cpu_count() - 1 if self.max_processes == -1 else self.max_processes,
                self.n_trajectories,
            )

            manager = mp.Manager()
            init_lock = manager.Lock()
            id_counter = manager.Value("i", 0)

            print(
                f"Initializing persistent worker pool with {self._n_workers} workers..."
            )

            planner_cpu = deepcopy(self.planner).to(device=torch.device("cpu"))

            self._pool = mp.Pool(
                processes=self._n_workers,
                initializer=_worker_init,
                initargs=(
                    planner_cpu,
                    self.planner.tensor_args["device"],
                    id_counter,
                    init_lock,
                    self.seed,
                ),
            )
            while id_counter.value < self._n_workers:
                sleep(1)

            print(f"Worker pool initialized.")

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        if not self.use_parallel:
            self.planner.reset(start_pos, goal_pos)
            return

        start_pos_cpu = start_pos.detach().cpu()
        goal_pos_cpu = goal_pos.detach().cpu()

        reset_commands = [
            {"type": "reset", "start_pos": start_pos_cpu, "goal_pos": goal_pos_cpu}
            for _ in range(self._n_workers)
        ]

        results = list(
            self._pool.imap_unordered(_worker_process_command, reset_commands)
        )

        ok = all(r.get("status") == "reset_done" for r in results)
        if not ok:
            print(f"Warning: Not all workers reset successfully")

    def optimize(self, **kwargs) -> List[torch.Tensor]:
        if self.use_parallel:
            return self._optimize_parallel(**kwargs)
        else:
            return self._optimize_sequential(**kwargs)

    def _optimize_parallel(self, **kwargs) -> List[torch.Tensor]:
        commands = [
            {"type": "optimize", "traj_id": i, "kwargs": kwargs}
            for i in range(self.n_trajectories)
        ]

        print(
            f"\nStarting parallel optimization: {self.n_trajectories} trajectories with {self._n_workers} workers..."
        )

        trajectories = []
        for response in self._pool.imap_unordered(_worker_process_command, commands):
            if response.get("status") == "success":
                result = response["result"].to(**self.planner.tensor_args)
                trajectories.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Completed: {len(trajectories)}/{self.n_trajectories} trajectories successful")
        return trajectories

    def _optimize_sequential(self, **kwargs) -> List[torch.Tensor]:
        trajectories = []
        for i in range(self.n_trajectories):
            try:
                kwargs["traj_id"] = i
                self.planner.planner_id = 0
                traj = self.planner.optimize(**kwargs)
                trajectories.append(traj)
            except Exception as e:
                print(f"Trajectory {i} optimization failed: {e}")
                traceback.print_exc()
        return trajectories

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.shutdown()
