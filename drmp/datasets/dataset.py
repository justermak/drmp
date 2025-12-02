import os
import traceback

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from drmp.config import N_DIMS
from drmp.datasets.normalization import Normalizer, get_normalizers
from drmp.planning.metrics import compute_free_fraction, compute_collision_intensity, compute_success
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.hybrid_planner import HybridPlanner
from drmp.planning.planners.parallel_sample_based_planner import (
    ParallelSampleBasedPlanner,
)
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.yaml import save_config_to_yaml
from drmp.utils.visualizer import Visualizer
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
        cutoff_margin: float,
        tensor_args: dict,
    ):
        
        self.tensor_args: dict = tensor_args
        self.env_name: str = env_name
        self.env: EnvBase = ENVS[env_name](tensor_args=tensor_args)
        self.robot: Robot = Robot(margin=robot_margin, tensor_args=tensor_args)
        self.generating_robot: Robot = Robot(margin=robot_margin + cutoff_margin, tensor_args=tensor_args)
        self.normalizer_name: str = normalizer_name
        self.normalizer: Normalizer = NORMALIZERS[self.normalizer_name]()
        self.datasets_dir: str = datasets_dir
        self.dataset_name: str = dataset_name
        self.dataset_dir: str = os.path.join(datasets_dir, dataset_name)
        self.trajs: torch.Tensor = torch.empty(0)
        self.start_states: torch.Tensor = torch.empty(0)
        self.goal_states: torch.Tensor = torch.empty(0)
        self.trajs_normalized: torch.Tensor = torch.empty(0)
        self.start_states_normalized: torch.Tensor = torch.empty(0)
        self.goal_states_normalized: torch.Tensor = torch.empty(0)
        self.n_trajs: int = 0
        self.n_support_points: int = 0
        self.state_dim: int = 0

    def load_data(self) -> None:
        trajs_free: list[torch.Tensor] = []
        n_trajs: int = 0
        for current_dir, _, files in os.walk(self.dataset_dir, topdown=True):
            if "trajs-free.pt" in files:
                trajs_free_part = torch.load(
                    os.path.join(current_dir, "trajs-free.pt"),
                    map_location=self.tensor_args["device"],
                )
                n_trajs += len(trajs_free_part)
                trajs_free.append(trajs_free_part)

        trajs_free = torch.cat(trajs_free, dim=0)

        self.trajs = trajs_free
        self.start_states = self.trajs[..., 0, :]
        self.goal_states = self.trajs[..., -1, :]
        self.n_trajs, self.n_support_points, self.state_dim = self.trajs.shape
        
        normalizers_loaded = self.load_normalizers_state()
        
        if not normalizers_loaded:
            self.normalizer.fit(self.trajs)
            self.save_normalizer_state()
        
        self.trajs_normalized = self.normalizer.normalize(self.trajs)
        self.start_states_normalized = self.normalizer.normalize(self.start_states)
        self.goal_states_normalized = self.normalizer.normalize(self.goal_states)

    def __len__(self) -> int:
        return self.n_trajs

    def __getitem__(self, idx: int) -> dict:
        data: dict = {
            "trajs_normalized": self.trajs_normalized[idx],
            "start_states_normalized": self.start_states_normalized[idx],
            "goal_states_normalized": self.goal_states_normalized[idx],
        }

        return data

    def save_normalizer_state(self) -> None:
        normalizers_path = os.path.join(self.dataset_dir, "normalizer_state.pt")
        
        normalizers_state = {
            "mins": self.normalizer.mins,
            "maxs": self.normalizer.maxs,
            "range": self.normalizer.range,
            "constant_mask": self.normalizer.constant_mask,
        }
        
        torch.save(normalizers_state, normalizers_path)
        print(f"Saved normalizers state to {normalizers_path}")

    def load_normalizers_state(self) -> bool:
        normalizers_path = os.path.join(self.dataset_dir, "normalizer_state.pt")
        
        if not os.path.exists(normalizers_path):
            return False
        
        normalizers_state = torch.load(
            normalizers_path,
            map_location=self.tensor_args["device"],
        )
        
        self.normalizer.mins = normalizers_state["mins"]
        self.normalizer.maxs = normalizers_state["maxs"]
        self.normalizer.range = normalizers_state["range"]
        self.normalizer.constant_mask = normalizers_state["constant_mask"]
        
        
        print(f"Loaded normalizers state from {normalizers_path}")
        return True

    def generate_trajectories_for_task(
        self,
        id: str,
        n_trajectories: int,
        threshold_start_goal_pos: float,
        sample_iters: int,
        opt_iters: int,
        n_support_points: int,
        duration: float,
        debug: bool,
    ) -> tuple[int, int]:
        start_pos, goal_pos, success = self.env.random_collision_free_start_goal(robot=self.generating_robot, n_samples=1, threshold_start_goal_pos=threshold_start_goal_pos)
        if not success:
            print("Could not find sufficient collision-free start/goal pairs for test tasks, try reducing the threshold, robot margin or object density")
            return 
        
        start_pos = start_pos.squeeze(0)
        goal_pos = goal_pos.squeeze(0)
        
        
        rrt_connect_default_params_env = self.env.get_rrt_connect_params()
        sample_based_planner = RRTConnect(
            env=self.env,
            robot=self.generating_robot,
            start_pos=start_pos,
            goal_pos=goal_pos,
            tensor_args=self.tensor_args,
            step_size=rrt_connect_default_params_env["step_size"],
            n_radius=rrt_connect_default_params_env["n_radius"],
            n_pre_samples=rrt_connect_default_params_env["n_pre_samples"],
        )

        parallel_sample_based_planner = ParallelSampleBasedPlanner(
            planner=sample_based_planner,
            n_trajectories=n_trajectories,
            use_parallel=True,
            max_processes=-1,
        )

        gpmp_default_params_env = self.env.get_gpmp2_params()
        optimization_based_planner = GPMP2(
            robot=self.generating_robot,
            n_dof=N_DIMS,
            num_particles_per_goal=n_trajectories,
            start_pos=start_pos,
            multi_goal_pos=goal_pos.unsqueeze(0),
            env=self.env,
            tensor_args=self.tensor_args,
            n_support_points=n_support_points,
            dt=duration / (n_support_points - 1),
            num_samples=gpmp_default_params_env["num_samples"],
            sigma_start=gpmp_default_params_env["sigma_start"],
            sigma_gp=gpmp_default_params_env["sigma_gp"],
            sigma_goal_prior=gpmp_default_params_env["sigma_goal_prior"],
            sigma_coll=gpmp_default_params_env["sigma_coll"],
            step_size=gpmp_default_params_env["step_size"],
            delta=gpmp_default_params_env["delta"],
            method=gpmp_default_params_env["method"],
        )

        planner = HybridPlanner(
            sample_based_planner=parallel_sample_based_planner, 
            opt_based_planner=optimization_based_planner, 
            tensor_args=self.tensor_args
        )

        trajs = planner.optimize(
            sample_iters=sample_iters,
            opt_iters=opt_iters,
            debug=debug,
        )

        # Double the dataset by adding inverted trajectories
        trajs_inverted = self.robot.invert_trajectories(trajs=trajs)
        trajs = torch.cat([trajs, trajs_inverted], dim=0)
        
        trajs_collision, trajs_free, points_collision_mask = self.env.get_trajs_collision_and_free(robot=self.robot, trajs=trajs)

        print(f"--------- STATISTICS ---------")
        print(f"total trajectories: {len(trajs)}")
        print(f"free fraction: {compute_free_fraction(trajs_free, trajs_collision) * 100:.2f}")
        print(f"collision intensity: {compute_collision_intensity(points_collision_mask) * 100:.2f}")
        print(f"success {compute_success(trajs_free)}")

        results_dir = os.path.join(self.dataset_dir, id)
        os.makedirs(results_dir, exist_ok=True)
        torch.save(trajs_collision, os.path.join(results_dir, f"trajs-collision.pt"))
        torch.save(trajs_free, os.path.join(results_dir, f"trajs-free.pt"))

        planning_visualizer = Visualizer(
            env=self.env,
            robot=self.robot,
        )

        planning_visualizer.render_scene(
            trajs=trajs,
            start_state=start_pos,
            goal_state=goal_pos,
            save_path=os.path.join(results_dir, f"trajectories_figure.png")
        )

        n_trajs_collision, n_trajs_free = len(trajs_collision), len(trajs_free)

        return n_trajs_collision, n_trajs_free

    def generate_data(
        self,
        n_tasks: int,
        n_trajectories: int,
        threshold_start_goal_pos: float,
        sample_iters: int,
        opt_iters: int,
        n_support_points: int,
        duration: float,
        val_portion: float,
        debug: bool,
    ) -> None:
        
        os.makedirs(self.dataset_dir, exist_ok=True)

        config: dict = {
            "env_name": self.env_name,
            "datasets_dir": self.datasets_dir,
            "dataset_name": self.dataset_name,
            "dataset_dir": self.dataset_dir,
            "normalizer_name": self.normalizer_name,
            "n_tasks": n_tasks,
            "n_trajectories": n_trajectories,
            "threshold_start_goal_pos": threshold_start_goal_pos,
            "robot_margin": self.robot.margin,
            "cutoff_margin": self.generating_robot.margin - self.robot.margin,
            "sample_iters": sample_iters,
            "opt_iters": opt_iters,
            "n_support_points": n_support_points,
            "duration": duration,
            "val_portion": val_portion,
            "debug": debug,
        }

        save_config_to_yaml(config, os.path.join(self.dataset_dir, "config.yaml"))

        n_free_trajs: int = 0
        n_coll_trajs: int = 0
        n_failed_tasks: int = 0
        print(f"{'=' * 80}")
        print(f"Starting trajectory generation for {n_tasks} tasks")
        print(f"{'=' * 80}\n")

        for i in range(n_tasks):
            id = str(i)
            print(f"Task {i + 1}/{n_tasks}")
            print("=" * 80)

            try:
                n_coll, n_free = self.generate_trajectories_for_task(
                    id=id,
                    n_trajectories=n_trajectories,
                    threshold_start_goal_pos=threshold_start_goal_pos,
                    sample_iters=sample_iters,
                    opt_iters=opt_iters,
                    n_support_points=n_support_points,
                    duration=duration,
                    debug=debug,
                )
                n_free_trajs += n_free
                n_coll_trajs += n_coll
                if n_free == 0:
                    print(f"Found no collision-free trajectories for task {id}")
                    return 
                else:
                    print(f"Free trajectories: {n_free}")
                    print(f"Collision trajectories: {n_coll}")

            except Exception as e:
                n_failed_tasks += 1
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"Task {id} failed with error: {error_msg}")
                traceback.print_exc()

                print("Failed tasks:", n_failed_tasks)
                raise

        train, val = random_split(range(n_free_trajs), [1 - val_portion, val_portion])
        torch.save(
            train.indices,
            os.path.join(self.dataset_dir, f"train_idx.pt"),
        )
        torch.save(val.indices, os.path.join(self.dataset_dir, f"val_idx.pt"))
        
        print("\nLoading generated data to compute and save normalizers...")
        self.load_data()
        self.save_normalizer_state()

    def load_train_val_split(
        self, batch_size: int = 1
    ) -> tuple[Subset, DataLoader, Subset, DataLoader]:
        train_idx = torch.load(
            os.path.join(self.dataset_dir, f"train_idx.pt")
        )
        val_idx = torch.load(
            os.path.join(self.dataset_dir, f"val_idx.pt")
        )
        train_subset = Subset(self, train_idx)
        val_subset = Subset(self, val_idx)
        train_dataloader = DataLoader(train_subset, batch_size=batch_size)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size)

        return train_subset, train_dataloader, val_subset, val_dataloader
