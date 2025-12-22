import os
import pickle
from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference.runner_config import BaseRunnerConfig, BaseRunnerModelWrapper
from drmp.planning.metrics import (
    compute_collision_intensity,
    compute_free_fraction,
    compute_path_length,
    compute_sharpness,
    compute_success,
    compute_waypoints_variance,
)
from drmp.utils.torch_timer import TimerCUDA
from drmp.utils.visualizer import Visualizer
from drmp.utils.yaml import save_config_to_yaml


def run_inference_for_task(
    task_id: int,
    dataset: TrajectoryDataset,
    data_normalized: dict,
    n_samples: int,
    model_wrapper: BaseRunnerModelWrapper,
) -> Dict[str, Any]:
    robot = dataset.robot
    env = dataset.env
    with TimerCUDA() as timer_model_sampling:
        trajectories_iters, trajectories_final = model_wrapper.sample(
            dataset=dataset,
            data_normalized=data_normalized,
            n_samples=n_samples,
        )
    task_time = timer_model_sampling.elapsed

    use_extra_objects = model_wrapper.use_extra_objects
    trajectories_final_collision, trajectories_final_free, points_final_collision_mask = (
        env.get_trajectories_collision_and_free(trajectories=trajectories_final, robot=robot, on_extra=use_extra_objects)
    )

    success = compute_success(trajectories_final_free)
    free_fraction = compute_free_fraction(trajectories_final_free, trajectories_final_collision)
    collision_intensity = compute_collision_intensity(points_final_collision_mask).cpu().numpy()
    sharpness = None
    path_length = None
    waypoints_variance = None
    best_traj_idx = None
    traj_final_best = None
    path_length_best = None

    if trajectories_final_free is not None and trajectories_final_free.shape[0] > 0:
        sharpness = compute_sharpness(trajectories_final_free, robot)
        path_length = compute_path_length(trajectories_final_free, robot)
        waypoints_variance = (
            compute_waypoints_variance(trajectories_final_free, robot).cpu().numpy()
        )
        best_traj_idx = torch.argmin(path_length).item()
        traj_final_best = trajectories_final_free[best_traj_idx]
        path_length_best = torch.min(path_length).item()
        sharpness = sharpness.cpu().numpy()
        path_length = path_length.cpu().numpy()

    task_results = {
        "task_id": task_id,
        "start_pos": robot.get_position(
            dataset.normalizer.unnormalize(data_normalized["start_states_normalized"]).cpu().numpy()
        ),
        "goal_pos": robot.get_position(
            dataset.normalizer.unnormalize(data_normalized["goal_states_normalized"]).cpu().numpy()
        ),
        "n_samples": n_samples,
        "trajectories_iters": trajectories_iters,
        "trajectories_final": trajectories_final,
        "trajectories_final_collision": trajectories_final_collision,
        "trajectories_final_free": trajectories_final_free,
        "success": success,
        "free_fraction": free_fraction,
        "collision_intensity": collision_intensity,
        "best_traj_idx": best_traj_idx,
        "traj_final_best": traj_final_best,
        "path_length_best": path_length_best,
        "sharpness": sharpness,
        "path_length": path_length,
        "waypoints_variance": waypoints_variance,
        "t_task": task_time,
    }
    return task_results


def run_inference_on_dataset(
    subset: Subset,
    n_tasks: int,
    n_samples: int,
    model_wrapper: BaseRunnerModelWrapper,
) -> List[Dict[str, Any]]:
    results = []
    dataset: TrajectoryDataset = subset.dataset

    for i in tqdm(range(n_tasks), desc="Processing tasks"):
        idx = np.random.choice(subset.indices)
        data_normalized = dataset[idx]

        task_results = run_inference_for_task(
            task_id=i,
            dataset=dataset,
            data_normalized=data_normalized,
            n_samples=n_samples,
            model_wrapper=model_wrapper,
        )
        results.append(task_results)

    return results


def compute_stats(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(results) == 0:
        return None

    n_tasks = len(results)
    n_samples = results[0]["n_samples"]
    success = [r["success"] for r in results]
    free_fraction = [r["free_fraction"] for r in results]
    collision_intensity = [r["collision_intensity"] for r in results]
    t = [r["t_task"] for r in results]
    path_length_best = [
        r["path_length_best"]
        for r in results
        if r["path_length_best"] is not None
    ]
    sharpness = [
        r["sharpness"].mean().item()
        for r in results
        if r["sharpness"] is not None
    ]
    path_length = [
        r["path_length"].mean().item()
        for r in results
        if r["path_length"] is not None
    ]
    waypoints_variance = [
        r["waypoints_variance"]
        for r in results
        if r["waypoints_variance"] is not None
    ]

    stats = {
        "n_tasks": n_tasks,
        "n_samples": n_samples,
        "time_avg": np.mean(t),
        "time_std": np.std(t),
        "success_rate_avg": np.mean(success),
        "success_rate_std": np.std(success),
        "free_fraction_avg": np.mean(free_fraction),
        "free_fraction_std": np.std(free_fraction),
        "collision_intensity_avg": np.mean(collision_intensity),
        "collision_intensity_std": np.std(collision_intensity),
        "path_length_best_avg": np.mean(path_length_best) if path_length_best else None,
        "path_length_best_std": np.std(path_length_best) if path_length_best else None,
        "sharpness_avg": np.mean(sharpness) if sharpness else None,
        "sharpness_std": np.std(sharpness) if sharpness else None,
        "path_length_avg": np.mean(path_length) if path_length else None,
        "path_length_std": np.std(path_length) if path_length else None,
        "waypoints_variance_avg": np.mean(waypoints_variance) if waypoints_variance else None,
        "waypoints_variance_std": np.std(waypoints_variance) if waypoints_variance else None,
    }
    return stats


def print_stats(results):
    print("=" * 80)

    for split_name in ["train", "val", "test"]:
        stats = results.get(f"{split_name}_stats")
        if stats is None:
            continue
        print(f"-------- {split_name.upper()} SPLIT --------")
        print(f"n_tasks: {stats['n_tasks']}")
        print(f"n_samples: {stats['n_samples']}")
        print(f"Time to generate n_samples: {stats['time_avg']:.3f} ± {2 * stats['time_std']:.3f} sec")
        print(f"Success rate: {stats['success_rate_avg'] * 100:.2f} ± {2 * stats['success_rate_std'] * 100:.2f}%")
        print(f"Free fraction: {stats['free_fraction_avg'] * 100:.2f} ± {2 * stats['free_fraction_std'] * 100:.2f}%")
        print(
            f"Collision intensity: {stats['collision_intensity_avg'] * 100:.2f} ± {2 * stats['collision_intensity_std'] * 100:.2f} %"
        )
        if stats["path_length_best_avg"] is not None:
            print(
                f"Best path length: {stats['path_length_best_avg']:.4f} ± {stats['path_length_best_std']:.4f}"
            )
        if stats["sharpness_avg"] is not None:
            print(f"Sharpness: {stats['sharpness_avg']:.4f} ± {stats['sharpness_std']:.4f}")
        if stats["path_length_avg"] is not None:
            print(f"Path length: {stats['path_length_avg']:.4f} ± {stats['path_length_std']:.4f}")
        if stats["waypoints_variance_avg"] is not None:
            print(f"Waypoints variance: {stats['waypoints_variance_avg']:.4f} ± {stats['waypoints_variance_std']:.4f}")


def visualize_results(
    results: Dict[str, List[Dict[str, Any]]],
    dataset: TrajectoryDataset,
    generation_dir: str,
):
    planner_visualizer = Visualizer(env=dataset.env, robot=dataset.robot)

    first_task = (
        results["test"][0]
        if "test" in results and len(results["test"]) > 0
        else (
            results["val"][0]
            if "val" in results and len(results["val"]) > 0
            else results["train"][0]
        )
    )
    start_pos = first_task["start_pos"]
    goal_pos = first_task["goal_pos"]
    trajectories_iters = first_task["trajectories_iters"]
    trajectories_final = first_task["trajectories_final"]
    best_traj_idx = first_task["best_traj_idx"]

    planner_visualizer.render_scene(
        trajectories=trajectories_final,
        best_traj_idx=best_traj_idx,
        start_state=start_pos,
        goal_state=goal_pos,
        save_path=os.path.join(generation_dir, "task0-trajectories.png"),
    )

    planner_visualizer.animate_robot_motion(
        trajectories=trajectories_final,
        best_traj_idx=best_traj_idx,
        start_state=start_pos,
        goal_state=goal_pos,
        save_path=os.path.join(generation_dir, f"task0-robot-motion.mp4"),
        n_frames=min(60, trajectories_final.shape[1]),
    )

    planner_visualizer.animate_optimization_iterations(
        trajectories=trajectories_iters,
        best_traj_idx=best_traj_idx,
        start_state=start_pos,
        goal_state=goal_pos,
        save_path=os.path.join(generation_dir, f"task0-opt-iters.mp4"),
        n_frames=min(60, len(trajectories_iters)),
    )

def create_test_subset(
    dataset: TrajectoryDataset,
    n_tasks: int,
    threshold_start_goal_pos: float,
    tensor_args: Dict[str, Any],
) -> Optional[Subset]:
    start_pos, goal_pos, success = dataset.env.random_collision_free_start_goal(
        robot=dataset.robot,
        n_samples=n_tasks,
        threshold_start_goal_pos=threshold_start_goal_pos,
    )
    if not success:
        print(
            "Could not find sufficient collision-free start/goal pairs for test tasks, "
            "try reducing the threshold, robot margin or object density"
        )
        return None
    
    test_dataset = copy(dataset)
    test_dataset.n_trajs = n_tasks
    test_dataset.trajs_normalized = torch.empty((n_tasks,), **tensor_args)
    test_dataset.start_states = torch.cat(
        [start_pos, torch.zeros_like(start_pos)], dim=-1
    )
    test_dataset.goal_states = torch.cat(
        [goal_pos, torch.zeros_like(goal_pos)], dim=-1
    )
    test_dataset.start_states_normalized = test_dataset.normalizer.normalize(
        test_dataset.start_states
    )
    test_dataset.goal_states_normalized = test_dataset.normalizer.normalize(
        test_dataset.goal_states
    )
    return Subset(test_dataset, list(range(n_tasks)))


def run_inference(
    runner_config: BaseRunnerConfig,
    dataset: TrajectoryDataset,
    train_subset: Optional[Subset],
    val_subset: Optional[Subset],
    test_subset: Optional[Subset],
    generations_dir: str,
    experiment_name: str,
    n_tasks: int,
    n_samples: int,
    debug: bool,
    tensor_args: Dict[str, Any],
) -> Dict[str, Any]:
    generation_dir = os.path.join(generations_dir, experiment_name)
    os.makedirs(generation_dir, exist_ok=True)

    config_dict = runner_config.to_dict()
    config_dict["n_tasks"] = n_tasks
    config_dict["n_samples"] = n_samples
    save_config_to_yaml(config_dict, os.path.join(generation_dir, "config.yaml"))

    model_wrapper = runner_config.prepare(dataset=dataset, tensor_args=tensor_args, n_samples=n_samples)

    results = {}

    print('=' * 80)
    print(f"Starting trajectory generation for {n_tasks} tasks per split")

    if train_subset is not None:
        print('=' * 80)
        print("Processing TRAIN split...")
        results["train"] = run_inference_on_dataset(
            subset=train_subset,
            n_tasks=n_tasks,
            n_samples=n_samples,
            model_wrapper=model_wrapper,
        )
        results["train_stats"] = compute_stats(results["train"])
    if val_subset is not None:
        print('=' * 80)
        print("Processing VAL split...")
        results["val"] = run_inference_on_dataset(
            subset=val_subset,
            n_tasks=n_tasks,
            n_samples=n_samples,
            model_wrapper=model_wrapper,
        )
        results["val_stats"] = compute_stats(results["val"])
    if test_subset is not None:
        print('=' * 80)
        print("Processing TEST split...")
        results["test"] = run_inference_on_dataset(
            subset=test_subset,
            n_tasks=n_tasks,
            n_samples=n_samples,
            model_wrapper=model_wrapper,
        )
        results["test_stats"] = compute_stats(results["test"])

    print_stats(results)

    with open(os.path.join(generation_dir, "results_data_dict.pickle"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    if debug:
        visualize_results(
            results=results,
            dataset=dataset,
            generation_dir=generation_dir,
        )

    return results
