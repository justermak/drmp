import os
import traceback
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.autonotebook import tqdm

from drmp.config import N_DIM
from drmp.datasets.filtering import get_filter_functions
from drmp.datasets.normalization import Normalizer, get_normalizers
from drmp.planning.metrics import (
    compute_collision_intensity,
    compute_free_fraction,
    compute_success,
)
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.hybrid_planner import HybridPlanner
from drmp.planning.planners.parallel_sample_based_planner import (
    ParallelSampleBasedPlanner,
)
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.visualizer import Visualizer
from drmp.utils.yaml import save_config_to_yaml
from drmp.world.environments import EnvBase, get_envs
from drmp.world.robot import Robot

NORMALIZERS = get_normalizers()

ENVS = get_envs()


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        datasets_dir: str,
        dataset_name: str,
        env_name: str,
        normalizer_name: str,
        robot_margin: float,
        generating_robot_margin: float,
        n_support_points: int,
        duration: float,
        apply_augmentations: bool,
        tensor_args: Dict[str, Any],
    ):
        self.tensor_args = tensor_args
        self.env_name = env_name
        self.n_support_points = n_support_points
        self.duration: float = duration
        self.robot_margin = robot_margin
        self.generating_robot_margin = generating_robot_margin
        self.apply_augmentations = apply_augmentations
        self.env: EnvBase = ENVS[env_name](tensor_args=tensor_args)
        self.robot: Robot = Robot(
            margin=robot_margin,
            dt=duration / (n_support_points - 1),
            tensor_args=tensor_args,
        )
        self.generating_robot: Robot = Robot(
            margin=generating_robot_margin,
            dt=duration / (n_support_points - 1),
            tensor_args=tensor_args,
        )
        self.normalizer_name = normalizer_name
        self.normalizer: Normalizer = NORMALIZERS[self.normalizer_name]()
        self.datasets_dir = datasets_dir
        self.dataset_name = dataset_name
        self.dataset_dir: str = os.path.join(datasets_dir, dataset_name)
        self.trajectories: torch.Tensor = torch.empty(0)
        self.start_states: torch.Tensor = torch.empty(0)
        self.goal_states: torch.Tensor = torch.empty(0)
        self.trajectories_normalized: torch.Tensor = torch.empty(0)
        self.start_states_normalized: torch.Tensor = torch.empty(0)
        self.goal_states_normalized: torch.Tensor = torch.empty(0)
        self.n_trajectories: int = 0
        self.state_dim: int = 0

    def load_data(self) -> None:
        print(self.dataset_dir)
        trajectories_free: list[torch.Tensor] = []
        n_trajectories: int = 0
        for current_dir, _, files in os.walk(self.dataset_dir, topdown=True):
            if "trajectories-free.pt" in files:
                trajectories_free_part = torch.load(
                    os.path.join(current_dir, "trajectories-free.pt"),
                    map_location=self.tensor_args["device"],
                )
                n_trajectories += len(trajectories_free_part)
                trajectories_free.append(trajectories_free_part)

        trajectories_free = torch.cat(trajectories_free, dim=0)

        self.trajectories = trajectories_free
        self.start_states = self.trajectories[..., 0, :]
        self.goal_states = self.trajectories[..., -1, :]
        self.n_trajectories, self.n_support_points, self.state_dim = (
            self.trajectories.shape
        )

        self.normalizer.fit(self.trajectories)

        self.trajectories_normalized = self.normalizer.normalize(self.trajectories)
        self.start_states_normalized = self.normalizer.normalize(self.start_states)
        self.goal_states_normalized = self.normalizer.normalize(self.goal_states)

    def __len__(self) -> int:
        return self.n_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories_normalized[idx]
        start_state = self.start_states_normalized[idx]
        goal_state = self.goal_states_normalized[idx]
        
        if self.apply_augmentations and torch.rand(1).item() < 0.5:
            trajectory = torch.flip(trajectory, dims=[0])
            start_state, goal_state = goal_state, start_state
        
        data = {
            "trajectories_normalized": trajectory,
            "start_states_normalized": start_state,
            "goal_states_normalized": goal_state,
        }

        return data

    def _create_planner(
        self,
        n_trajectories: int,
        n_support_points: int,
        use_parallel: bool,
        max_processes: int,
        rrt_connect_step_size: float,
        rrt_connect_n_radius: float,
        rrt_connect_n_samples: int,
        gpmp2_n_interpolate: int,
        gpmp2_num_samples: int,
        gpmp2_sigma_start: float,
        gpmp2_sigma_goal_prior: float,
        gpmp2_sigma_gp: float,
        gpmp2_sigma_collision: float,
        gpmp2_step_size: float,
        gpmp2_delta: float,
        gpmp2_method: str,
        seed: int,
    ) -> HybridPlanner:
        sample_based_planner = RRTConnect(
            env=self.env,
            robot=self.generating_robot,
            tensor_args=self.tensor_args,
            step_size=rrt_connect_step_size,
            n_radius=rrt_connect_n_radius,
            n_samples=rrt_connect_n_samples,
        )

        parallel_sample_based_planner = ParallelSampleBasedPlanner(
            planner=sample_based_planner,
            n_trajectories=n_trajectories,
            use_parallel=use_parallel,
            max_processes=max_processes,
            seed=seed,
        )

        optimization_based_planner = GPMP2(
            robot=self.generating_robot,
            n_dof=N_DIM,
            n_trajectories=n_trajectories,
            env=self.env,
            tensor_args=self.tensor_args,
            n_support_points=n_support_points,
            dt=self.generating_robot.dt,
            n_interpolate=gpmp2_n_interpolate,
            num_samples=gpmp2_num_samples,
            sigma_start=gpmp2_sigma_start,
            sigma_gp=gpmp2_sigma_gp,
            sigma_goal_prior=gpmp2_sigma_goal_prior,
            sigma_collision=gpmp2_sigma_collision,
            step_size=gpmp2_step_size,
            delta=gpmp2_delta,
            method=gpmp2_method,
        )

        return HybridPlanner(
            sample_based_planner=parallel_sample_based_planner,
            opt_based_planner=optimization_based_planner,
            tensor_args=self.tensor_args,
        )

    def generate_trajectories_for_task(
        self,
        id: str,
        planner: HybridPlanner,
        threshold_start_goal_pos: float,
        sample_steps: int,
        opt_steps: int,
        debug: bool,
    ) -> Tuple[int, int]:
        start_pos, goal_pos, success = self.env.random_collision_free_start_goal(
            robot=self.generating_robot,
            n_samples=1,
            threshold_start_goal_pos=threshold_start_goal_pos,
        )
        if not success:
            print(
                "Could not find sufficient collision-free start/goal pairs for test tasks"
            )
            return

        start_pos = start_pos.squeeze(0)
        goal_pos = goal_pos.squeeze(0)

        planner.reset(start_pos, goal_pos)

        trajectories = planner.optimize(
            sample_steps=sample_steps,
            opt_steps=opt_steps,
            debug=debug,
        )

        trajectories_collision, trajectories_free, points_collision_mask = (
            self.env.get_trajectories_collision_and_free(
                robot=self.robot, trajectories=trajectories
            )
        )

        print("--------- STATISTICS ---------")
        print(f"total trajectories: {len(trajectories)}")
        print(
            f"free fraction: {compute_free_fraction(trajectories_free, trajectories_collision) * 100:.2f}"
        )
        print(
            f"collision intensity: {compute_collision_intensity(points_collision_mask) * 100:.2f}"
        )
        print(f"success {compute_success(trajectories_free)}")

        results_dir = os.path.join(self.dataset_dir, id)
        os.makedirs(results_dir, exist_ok=True)
        torch.save(
            trajectories_collision,
            os.path.join(results_dir, "trajectories-collision.pt"),
        )
        torch.save(trajectories_free, os.path.join(results_dir, "trajectories-free.pt"))

        if debug:
            planning_visualizer = Visualizer(
                env=self.env,
                robot=self.robot,
            )

            planning_visualizer.render_scene(
                trajectories=trajectories,
                start_state=start_pos,
                goal_state=goal_pos,
                save_path=os.path.join(results_dir, "trajectories_figure.png"),
            )

        n_trajectories_collision, n_trajectories_free = (
            len(trajectories_collision),
            len(trajectories_free),
        )

        return n_trajectories_collision, n_trajectories_free

    def filter_data(
        self,
        train_idxs: List[int],
        val_idxs: List[int],
        task_start_idxs: torch.Tensor,
        filtering_config: Dict[str, Any],
    ) -> Tuple[List[int], List[int]]:
        assert len(self.trajectories) > 0, "Trajectories must be loaded before filtering"
        
        if not filtering_config:
            return train_idxs, val_idxs
        
        filter_functions = get_filter_functions()
        indices_to_exclude = set()
        
        for filter_name, filter_params in filtering_config.items():
            if filter_name not in filter_functions:
                print(f"Warning: Unknown filter function '{filter_name}', skipping")
                continue
            
            filter_fn = filter_functions[filter_name]
            excluded = filter_fn(
                trajectories=self.trajectories,
                robot=self.robot,
                task_start_idxs=task_start_idxs,
                **filter_params
            )
            indices_to_exclude.update(excluded)
        
        train_filtered = [idx for idx in train_idxs if idx not in indices_to_exclude]
        val_filtered = [idx for idx in val_idxs if idx not in indices_to_exclude]
        
        return train_filtered, val_filtered

    def _scan_existing_tasks(self) -> Dict[int, Tuple[int, int]]:
        existing_tasks = {}
        if not os.path.exists(self.dataset_dir):
            return existing_tasks

        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                task_id = int(item)
                collision_path = os.path.join(item_path, "trajectories-collision.pt")
                free_path = os.path.join(item_path, "trajectories-free.pt")

                if os.path.exists(collision_path) and os.path.exists(free_path):
                    trajectories_coll = torch.load(collision_path)
                    trajectories_free = torch.load(free_path)
                    n_coll = len(trajectories_coll)
                    n_free = len(trajectories_free)
                    del trajectories_coll, trajectories_free

                    existing_tasks[task_id] = (n_coll, n_free)

        return existing_tasks

    def generate_data(
        self,
        n_tasks: int,
        n_trajectories: int,
        threshold_start_goal_pos: float,
        sample_steps: int,
        opt_steps: int,
        val_portion: float,
        use_parallel: bool,
        max_processes: int,
        seed: int,
        rrt_connect_step_size: float,
        rrt_connect_n_radius: float,
        rrt_connect_n_samples: int,
        gpmp2_n_interpolate: int,
        gpmp2_num_samples: int,
        gpmp2_sigma_start: float,
        gpmp2_sigma_goal_prior: float,
        gpmp2_sigma_gp: float,
        gpmp2_sigma_collision: float,
        gpmp2_step_size: float,
        gpmp2_delta: float,
        gpmp2_method: str,
        debug: bool,
    ) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)

        config: dict = {
            "env_name": self.env_name,
            "datasets_dir": self.datasets_dir,
            "dataset_name": self.dataset_name,
            "dataset_dir": self.dataset_dir,
            "normalizer_name": self.normalizer_name,
            "n_support_points": self.n_support_points,
            "duration": self.duration,
            "robot_margin": self.robot_margin,
            "generating_robot_margin": self.generating_robot_margin,
            "n_tasks": n_tasks,
            "n_trajectories": n_trajectories,
            "threshold_start_goal_pos": threshold_start_goal_pos,
            "sample_steps": sample_steps,
            "opt_steps": opt_steps,
            "val_portion": val_portion,
            "debug": debug,
        }

        save_config_to_yaml(config, os.path.join(self.dataset_dir, "config.yaml"))

        existing_tasks = self._scan_existing_tasks()
        if existing_tasks:
            print(f"Found {len(existing_tasks)} existing tasks, will resume from there")
            print(f"Existing task IDs: {sorted(existing_tasks.keys())}")

        planner = self._create_planner(
            n_trajectories=n_trajectories,
            n_support_points=self.n_support_points,
            use_parallel=use_parallel,
            max_processes=max_processes,
            rrt_connect_step_size=rrt_connect_step_size,
            rrt_connect_n_radius=rrt_connect_n_radius,
            rrt_connect_n_samples=rrt_connect_n_samples,
            gpmp2_n_interpolate=gpmp2_n_interpolate,
            gpmp2_num_samples=gpmp2_num_samples,
            gpmp2_sigma_start=gpmp2_sigma_start,
            gpmp2_sigma_goal_prior=gpmp2_sigma_goal_prior,
            gpmp2_sigma_gp=gpmp2_sigma_gp,
            gpmp2_sigma_collision=gpmp2_sigma_collision,
            gpmp2_step_size=gpmp2_step_size,
            gpmp2_delta=gpmp2_delta,
            gpmp2_method=gpmp2_method,
            seed=seed + len(existing_tasks),
        )

        task_start_idxs = []
        n_free_trajectories: int = 0
        n_coll_trajectories: int = 0
        n_failed_tasks: int = 0
        n_skipped_tasks: int = 0

        print(f"{'=' * 80}")
        print(f"Starting trajectory generation for {n_tasks} tasks")
        print(f"{'=' * 80}\n")

        with tqdm(
            total=n_tasks,
            mininterval=1 if debug else 10,
            desc="Generating data",
        ) as pbar:
            for i in range(n_tasks):
                id = str(i)
                pbar.set_description(f"Task {i + 1}/{n_tasks}")

                try:
                    if i in existing_tasks:
                        n_coll, n_free = existing_tasks[i]
                        n_skipped_tasks += 1
                        pbar.set_postfix(
                            {
                                "status": "skipped",
                                "free": n_free,
                                "coll": n_coll,
                                "total_free": n_free_trajectories,
                                "skipped": n_skipped_tasks,
                            }
                        )
                        torch.manual_seed(seed + i)
                    else:
                        n_coll, n_free = self.generate_trajectories_for_task(
                            id=id,
                            planner=planner,
                            threshold_start_goal_pos=threshold_start_goal_pos,
                            sample_steps=sample_steps,
                            opt_steps=opt_steps,
                            debug=debug,
                        )

                        if n_free == 0:
                            print(f"Found no collision-free trajectories for task {id}")
                            return

                        pbar.set_postfix(
                            {
                                "status": "generated",
                                "free": n_free,
                                "coll": n_coll,
                                "total_free": n_free_trajectories,
                                "skipped": n_skipped_tasks,
                            }
                        )

                    task_start_idxs.append(n_free_trajectories)
                    n_free_trajectories += n_free
                    n_coll_trajectories += n_coll

                except Exception as e:
                    n_failed_tasks += 1
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    print(f"Task {id} failed with error: {error_msg}")
                    traceback.print_exc()

                    print("Failed tasks:", n_failed_tasks)
                    raise

                pbar.update(1)

            task_start_idxs.append(n_free_trajectories)
            planner.shutdown()

        task_start_idxs = torch.tensor(task_start_idxs, dtype=torch.long)
        train_tasks, val_tasks = random_split(
            range(n_tasks), [1 - val_portion, val_portion]
        )
        train_tasks_idxs = torch.tensor(train_tasks.indices)
        val_tasks_idxs = torch.tensor(val_tasks.indices)
        train_idxs = [
            i
            for task_start_idx, task_end_idx in zip(
                task_start_idxs[train_tasks_idxs], task_start_idxs[train_tasks_idxs + 1]
            )
            for i in range(task_start_idx, task_end_idx)
        ]
        val_idxs = [
            i
            for task_start_idx, task_end_idx in zip(
                task_start_idxs[val_tasks_idxs], task_start_idxs[val_tasks_idxs + 1]
            )
            for i in range(task_start_idx, task_end_idx)
        ]

        torch.save(train_idxs, os.path.join(self.dataset_dir, "train_idx.pt"))
        torch.save(val_idxs, os.path.join(self.dataset_dir, "val_idx.pt"))
        torch.save(task_start_idxs, os.path.join(self.dataset_dir, "task_start_idxs.pt"))

    def load_train_val_split(
        self,
        batch_size: int = 1,
        use_filtered_trajectories: bool = False,
        filtering_config: Dict[str, Any] = None,
    ) -> Tuple[Subset, DataLoader, Subset, DataLoader]:
        train_idx = torch.load(os.path.join(self.dataset_dir, "train_idx.pt"))
        val_idx = torch.load(os.path.join(self.dataset_dir, "val_idx.pt"))
        
        if use_filtered_trajectories:
            if filtering_config is None:
                raise ValueError(
                    "filtering_config must be provided when use_filtered_trajectories=True"
                )
            
            task_start_idxs = torch.load(
                os.path.join(self.dataset_dir, "task_start_idxs.pt")
            )
            
            print("\nApplying trajectory filters...")
            train_idx, val_idx = self.filter_data(
                train_idxs=train_idx,
                val_idxs=val_idx,
                task_start_idxs=task_start_idxs,
                filtering_config=filtering_config,
            )
            print(f"Train dataset size after filtering: {len(train_idx)}")
            print(f"Val dataset size after filtering: {len(val_idx)}")
        
        train_subset = Subset(self, train_idx)
        val_subset = Subset(self, val_idx)

        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size)

        return train_subset, train_dataloader, val_subset, val_dataloader
