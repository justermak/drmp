import os
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.autonotebook import tqdm

from drmp.datasets.filtering import get_filter_functions
from drmp.datasets.normalization import NormalizerBase, get_normalizers
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.hybrid_planner import HybridPlanner
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.trajectory_utils import fit_bsplines_to_trajectories
from drmp.utils.visualizer import Visualizer
from drmp.utils.yaml import save_config_to_yaml
from drmp.world.environments import EnvBase, get_envs
from drmp.world.robot import RobotBase, get_robots

NORMALIZERS = get_normalizers()

ENVS = get_envs()
ROBOTS = get_robots()


def _worker_process_task(args):
    (
        task_id,
        dataset_dir,
        env_name,
        robot_name,
        robot_margin,
        generating_robot_margin,
        n_support_points,
        duration,
        tensor_args,
        n_trajectories,
        threshold_start_goal_pos,
        sample_steps,
        opt_steps,
        seed,
        rrt_connect_max_step_size,
        rrt_connect_max_radius,
        rrt_connect_n_samples,
        gpmp2_n_interpolate,
        gpmp2_num_samples,
        gpmp2_sigma_start,
        gpmp2_sigma_goal_prior,
        gpmp2_sigma_gp,
        gpmp2_sigma_collision,
        gpmp2_step_size,
        gpmp2_delta,
        gpmp2_method,
        debug,
        grid_map_sdf_fixed,
        grid_map_sdf_extra,
    ) = args

    try:
        env: EnvBase = ENVS[env_name](
            tensor_args=tensor_args,
            grid_map_sdf_fixed=grid_map_sdf_fixed,
            grid_map_sdf_extra=grid_map_sdf_extra,
        )
        generating_robot = ROBOTS[robot_name](
            margin=generating_robot_margin,
            dt=duration / (n_support_points - 1),
            tensor_args=tensor_args,
        )
        robot = ROBOTS[robot_name](
            margin=robot_margin,
            dt=duration / (n_support_points - 1),
            tensor_args=tensor_args,
        )

        sample_based_planner = RRTConnect(
            env=env,
            robot=generating_robot,
            tensor_args=tensor_args,
            max_step_size=rrt_connect_max_step_size,
            max_radius=rrt_connect_max_radius,
            n_samples=rrt_connect_n_samples,
            n_trajectories=n_trajectories,
        )

        optimization_based_planner = GPMP2(
            robot=generating_robot,
            n_dof=robot.n_dim,
            n_trajectories=n_trajectories,
            env=env,
            tensor_args=tensor_args,
            n_support_points=n_support_points,
            dt=generating_robot.dt,
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

        planner = HybridPlanner(
            sample_based_planner=sample_based_planner,
            optimization_based_planner=optimization_based_planner,
            tensor_args=tensor_args,
        )

        torch.manual_seed(seed + task_id)

        start_pos, goal_pos, success = env.random_collision_free_start_goal(
            robot=generating_robot,
            n_samples=1,
            threshold_start_goal_pos=threshold_start_goal_pos,
        )

        if not success:
            return task_id, 0, 0, "failed_start_goal"

        start_pos = start_pos.squeeze(0)
        goal_pos = goal_pos.squeeze(0)

        planner.reset(start_pos, goal_pos)

        trajectories = planner.optimize(
            sample_steps=sample_steps,
            opt_steps=opt_steps,
            debug=debug,
        )

        _, trajectories_free, _ = env.get_trajectories_collision_and_free(
            robot=robot, trajectories=trajectories
        )
        n_free = len(trajectories_free)
        n_collision = len(trajectories) - n_free

        torch.save(
            trajectories.cpu(),
            os.path.join(dataset_dir, f"trajectories_{task_id}.pt"),
        )

        if debug:
            planning_visualizer = Visualizer(
                env=env, robot=robot, use_extra_objects=False
            )
            try:
                planning_visualizer.render_scene(
                    trajectories=trajectories,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    save_path=os.path.join(
                        dataset_dir, f"trajectories_figure_{task_id}.png"
                    ),
                )
            except Exception as e:
                print(f"Visualization failed for task {task_id}: {e}")

        return task_id, n_collision, n_free, "success"

    except Exception as e:
        print(f"Task {task_id} failed with error: {e}")
        traceback.print_exc()
        return task_id, 0, 0, str(e)


class TrajectoryDatasetBase(Dataset, ABC):
    def __init__(
        self,
        datasets_dir: str,
        dataset_name: str,
        env_name: str,
        robot_name: str,
        robot_margin: float,
        generating_robot_margin: float,
        n_support_points: int,
        duration: float,
        apply_augmentations: bool,
        tensor_args: Dict[str, Any],
    ):
        self.tensor_args = tensor_args
        self.env_name = env_name
        self.robot_name = robot_name
        self.n_support_points = n_support_points
        self.duration = duration
        self.robot_margin = robot_margin
        self.generating_robot_margin = generating_robot_margin
        self.apply_augmentations = apply_augmentations
        self.env: EnvBase = ENVS[env_name](tensor_args=tensor_args)
        self.robot: RobotBase = ROBOTS[robot_name](
            margin=robot_margin,
            dt=duration / (n_support_points - 1),
            tensor_args=tensor_args,
        )
        self.generating_robot: RobotBase = ROBOTS[robot_name](
            margin=generating_robot_margin,
            dt=duration / (n_support_points - 1),
            tensor_args=tensor_args,
        )
        self.datasets_dir = datasets_dir
        self.dataset_name = dataset_name
        self.dataset_dir: str = os.path.join(datasets_dir, dataset_name)
        self.trajectories: torch.Tensor = None
        self.start_pos: torch.Tensor = None
        self.goal_pos: torch.Tensor = None
        self.normalizer: NormalizerBase = None
        self.n_trajectories: int = 0

    @abstractmethod
    def load_data(self) -> None:
        print(self.dataset_dir)

        files = [
            f
            for f in os.listdir(self.dataset_dir)
            if f.startswith("trajectories_") and f.endswith(".pt")
        ]

        files_with_ids = []
        for f in files:
            try:
                id_ = int(f.split("_")[1].split(".")[0])
                files_with_ids.append((id_, f))
            except Exception:
                pass

        files_with_ids.sort(key=lambda x: x[0])

        trajectories_list = []
        for _, f in files_with_ids:
            t = torch.load(
                os.path.join(self.dataset_dir, f),
                map_location=self.tensor_args["device"],
            )
            trajectories_list.append(t)

        if not trajectories_list:
            print("No trajectories found!")
            return torch.empty(0)

        trajectories = torch.cat(trajectories_list, dim=0)

        if trajectories.numel() == 0:
            return

        self.n_trajectories, n_support_points, n_dim = (
            trajectories.shape
        )
        assert n_support_points == self.n_support_points and n_dim in (self.robot.n_dim, 2 * self.robot.n_dim)
        
        self.trajectories = trajectories[..., : self.robot.n_dim]
        self.start_pos = self.trajectories[..., 0, :self.robot.n_dim]
        self.goal_pos = self.trajectories[..., -1, :self.robot.n_dim]

    def __len__(self) -> int:
        return self.n_trajectories

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass

    def filter_data(
        self,
        train_idxs: List[int],
        val_idxs: List[int],
        task_start_idxs: torch.Tensor,
        filtering_config: Dict[str, Any],
    ) -> Tuple[List[int], List[int]]:
        assert len(self.trajectories) > 0, (
            "Trajectories must be loaded before filtering"
        )
        print(filtering_config)
        filter_functions = get_filter_functions()
        indices_to_exclude = set()
        for filter_name, filter_params in filtering_config.items():
            filter_fn = filter_functions[filter_name]
            excluded = filter_fn(
                trajectories=self.trajectories,
                robot=self.robot,
                env=self.env,
                task_start_idxs=task_start_idxs,
                **filter_params,
            )
            indices_to_exclude.update(excluded)

        train_filtered = [idx for idx in train_idxs if idx not in indices_to_exclude]
        val_filtered = [idx for idx in val_idxs if idx not in indices_to_exclude]

        return train_filtered, val_filtered

    def _scan_existing_tasks(self) -> Dict[int, int]:
        existing_tasks = {}
        if not os.path.exists(self.dataset_dir):
            return existing_tasks

        files = [
            f
            for f in os.listdir(self.dataset_dir)
            if f.startswith("trajectories_") and f.endswith(".pt")
        ]

        for f in files:
            try:
                task_id = int(f.split("_")[1].split(".")[0])
                trajectories = torch.load(
                    os.path.join(self.dataset_dir, f), map_location="cpu"
                )
                existing_tasks[task_id] = len(trajectories)
            except Exception:
                pass

        return existing_tasks

    def generate_data(
        self,
        n_tasks: int,
        n_trajectories: int,
        threshold_start_goal_pos: float,
        sample_steps: int,
        opt_steps: int,
        val_portion: float,
        seed: int,
        rrt_connect_max_step_size: float,
        rrt_connect_max_radius: float,
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
        max_processes: int,
    ) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)

        config: dict = {
            "env_name": self.env_name,
            "datasets_dir": self.datasets_dir,
            "dataset_name": self.dataset_name,
            "dataset_dir": self.dataset_dir,
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
            "max_processes": max_processes,
        }

        save_config_to_yaml(config, os.path.join(self.dataset_dir, "config.yaml"))

        existing_tasks = self._scan_existing_tasks()
        if existing_tasks:
            print(f"Found {len(existing_tasks)} existing tasks, will resume from there")
            print(f"Existing task IDs: {sorted(existing_tasks.keys())}")

        task_start_idxs = []

        grid_map_sdf_fixed = self.env.grid_map_sdf_fixed
        grid_map_sdf_extra = self.env.grid_map_sdf_extra

        tasks_to_run = []
        n_skipped_tasks = 0
        for i in range(n_tasks):
            if i in existing_tasks:
                n_skipped_tasks += 1
                continue

            task_args = (
                i,
                self.dataset_dir,
                self.env_name,
                self.robot_name,
                self.robot_margin,
                self.generating_robot_margin,
                self.n_support_points,
                self.duration,
                self.tensor_args,
                n_trajectories,
                threshold_start_goal_pos,
                sample_steps,
                opt_steps,
                seed,
                rrt_connect_max_step_size,
                rrt_connect_max_radius,
                rrt_connect_n_samples,
                gpmp2_n_interpolate,
                gpmp2_num_samples,
                gpmp2_sigma_start,
                gpmp2_sigma_goal_prior,
                gpmp2_sigma_gp,
                gpmp2_sigma_collision,
                gpmp2_step_size,
                gpmp2_delta,
                gpmp2_method,
                debug,
                grid_map_sdf_fixed,
                grid_map_sdf_extra,
            )
            tasks_to_run.append(task_args)

        print(f"{'=' * 80}")
        print(
            f"Starting trajectory generation for {n_tasks} tasks ({len(tasks_to_run)} new, {n_skipped_tasks} existing)"
        )
        print(f"Using {max_processes} processes")
        print(f"{'=' * 80}\n")

        ctx = mp.get_context("spawn")

        new_tasks = {}
        n_completed_tasks = 0
        n_failed_tasks = 0
        with tqdm(total=n_tasks, mininterval=1, desc="Generating data") as pbar:
            pbar.update(n_skipped_tasks)

            if max_processes > 1:
                with ctx.Pool(processes=max_processes) as pool:
                    for res in pool.imap_unordered(_worker_process_task, tasks_to_run):
                        n_completed_tasks += 1
                        task_id, n_collision, n_free, status = res
                        new_tasks[task_id] = (n_collision + n_free, status)

                        if status != "success" or n_free == 0:
                            n_failed_tasks += 1

                        if (
                            n_failed_tasks > 10
                            and n_failed_tasks > n_completed_tasks * 0.1
                        ):
                            raise RuntimeError(
                                f"Too many tasks with 0 free trajectories ({n_failed_tasks}/{pbar.n})"
                            )

                        pbar.set_postfix(
                            {
                                "status": status,
                                "collision": n_collision,
                                "free": n_free,
                                "failed": n_failed_tasks,
                            }
                        )
                        pbar.update(1)
            else:
                for args in tasks_to_run:
                    res = _worker_process_task(args)
                    n_completed_tasks += 1
                    task_id, n_collision, n_free, status = res
                    new_tasks[task_id] = (n_collision + n_free, status)

                    if status != "success" or n_free == 0:
                        n_failed_tasks += 1

                    if (
                            n_failed_tasks > 10
                            and n_failed_tasks > n_completed_tasks * 0.1
                        ):
                            raise RuntimeError(
                                f"Too many tasks with 0 free trajectories ({n_failed_tasks}/{pbar.n})"
                            )

                    pbar.update(1)

        for i in range(n_tasks):
            n_trajectories_i = 0
            if i in existing_tasks:
                n_trajectories_i = existing_tasks[i]
            elif i in new_tasks:
                n_trajectories_i, status = new_tasks[i]
                if status != "success":
                    n_trajectories_i = 0

            task_start_idxs.append(n_trajectories)
            n_trajectories += n_trajectories_i

        task_start_idxs.append(n_trajectories)

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
        torch.save(
            task_start_idxs, os.path.join(self.dataset_dir, "task_start_idxs.pt")
        )

    def load_train_val_split(
        self,
        batch_size: int = 1,
        filtering_config: Dict[str, Any] = {},
    ) -> Tuple[Subset, DataLoader, Subset, DataLoader]:
        train_idx = torch.load(os.path.join(self.dataset_dir, "train_idx.pt"))
        val_idx = torch.load(os.path.join(self.dataset_dir, "val_idx.pt"))

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


class TrajectoryDatasetDense(TrajectoryDatasetBase):
    def __init__(
        self,
        datasets_dir: str,
        dataset_name: str,
        env_name: str,
        robot_name: str,
        robot_margin: float,
        generating_robot_margin: float,
        n_support_points: int,
        duration: float,
        apply_augmentations: bool,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(
            datasets_dir=datasets_dir,
            dataset_name=dataset_name,
            env_name=env_name,
            robot_name=robot_name,
            robot_margin=robot_margin,
            generating_robot_margin=generating_robot_margin,
            n_support_points=n_support_points,
            duration=duration,
            apply_augmentations=apply_augmentations,
            tensor_args=tensor_args,
        )
        self.trajectories_normalized: torch.Tensor = torch.empty(0)
        self.start_pos_normalized: torch.Tensor = torch.empty(0)
        self.goal_pos_normalized: torch.Tensor = torch.empty(0)

    def load_data(self, normalizer_name="TrivialNormalizer") -> None:
        super().load_data()
        self.normalizer_name = normalizer_name
        self.normalizer: NormalizerBase = NORMALIZERS[self.normalizer_name]()
        self.normalizer.fit(self.trajectories)

        self.trajectories_normalized = self.normalizer.normalize(self.trajectories)
        self.start_pos_normalized = self.normalizer.normalize(self.start_pos)
        self.goal_pos_normalized = self.normalizer.normalize(self.goal_pos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories_normalized[idx]
        start_pos = self.start_pos_normalized[idx]
        goal_pos = self.goal_pos_normalized[idx]

        if self.apply_augmentations and torch.rand(1).item() < 0.5:
            trajectory = torch.flip(trajectory, dims=[0])
            start_pos, goal_pos = goal_pos, start_pos

        data = {
            "trajectories_normalized": trajectory,
            "start_pos_normalized": start_pos,
            "goal_pos_normalized": goal_pos,
        }

        return data


class TrajectoryDatasetBSpline(TrajectoryDatasetBase):
    def __init__(
        self,
        datasets_dir: str,
        dataset_name: str,
        env_name: str,
        robot_name: str,
        robot_margin: float,
        generating_robot_margin: float,
        n_support_points: int,
        duration: float,
        n_control_points: int,
        spline_degree: int,
        apply_augmentations: bool,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(
            datasets_dir=datasets_dir,
            dataset_name=dataset_name,
            env_name=env_name,
            robot_name=robot_name,
            robot_margin=robot_margin,
            generating_robot_margin=generating_robot_margin,
            n_support_points=n_support_points,
            duration=duration,
            apply_augmentations=apply_augmentations,
            tensor_args=tensor_args,
        )
        self.n_control_points = n_control_points
        self.real_n_control_points = n_control_points + 2 * (spline_degree - 1)
        self.spline_degree = spline_degree
        self.control_points: torch.Tensor = torch.empty(0)
        self.control_points_normalized: torch.Tensor = torch.empty(0)
        self.start_pos_normalized: torch.Tensor = torch.empty(0)
        self.goal_pos_normalized: torch.Tensor = torch.empty(0)

    def load_data(self, normalizer_name="TrivialNormalizer") -> None:
        super().load_data()

        print("Fitting B-Splines to trajectories...")
        self.control_points = fit_bsplines_to_trajectories(
            trajectories=self.trajectories,
            n_control_points=self.n_control_points,
            degree=self.spline_degree,
        )

        self.normalizer_name = normalizer_name
        self.normalizer: NormalizerBase = NORMALIZERS[self.normalizer_name]()
        self.normalizer.fit(self.control_points)

        self.control_points_normalized = self.normalizer.normalize(self.control_points)
        self.start_pos_normalized = self.normalizer.normalize(self.start_pos)
        self.goal_pos_normalized = self.normalizer.normalize(self.goal_pos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        control_points = self.control_points_normalized[idx]
        start_pos = self.start_pos_normalized[idx]
        goal_pos = self.goal_pos_normalized[idx]
        control_points_augmented = torch.cat([
            start_pos.unsqueeze(0).repeat(self.spline_degree - 1, 1),
            control_points,
            goal_pos.unsqueeze(0).repeat(self.spline_degree - 1, 1),   
        ], dim=-2)
        if self.apply_augmentations and torch.rand(1).item() < 0.5:
            control_points_augmented = torch.flip(control_points_augmented, dims=[0])
            start_pos, goal_pos = goal_pos, start_pos

        data = {
            "trajectories_normalized": control_points_augmented,
            "start_pos_normalized": start_pos,
            "goal_pos_normalized": goal_pos,
            "spline_degree": torch.tensor(self.spline_degree, dtype=torch.long),
        }

        return data
