import json
import os
import pickle
from copy import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from drmp.dataset.dataset import TrajectoryDataset
from drmp.planning.inference_config import ModelConfigBase, ModelWrapperBase
from drmp.planning.metrics import (
    bootstrap_confidence_interval,
    compute_collision_intensity,
    compute_free_fraction,
    compute_path_length,
    compute_sharpness,
    compute_success,
    compute_waypoints_variance,
)
from drmp.torch_timer import TimerCUDA
from drmp.visualizer import Visualizer


def run_inference_for_task(
    task_id: int,
    dataset: TrajectoryDataset,
    data: dict,
    n_trajectories_per_task: int,
    model_wrapper: ModelWrapperBase,
    return_full_data: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    robot = dataset.robot
    env = dataset.env
    with TimerCUDA() as timer_model_sampling:
        trajectories_iters, trajectories_final = model_wrapper.sample(
            dataset=dataset,
            data=data,
            n_trajectories_per_task=n_trajectories_per_task,
            debug=debug,
        )
    task_time = timer_model_sampling.elapsed
    if trajectories_final is None:
        return None
    (
        trajectories_final_collision,
        trajectories_final_free,
        points_final_collision_mask,
    ) = robot.get_trajectories_collision_and_free(
        env=env,
        trajectories=trajectories_final,
        on_extra=model_wrapper.use_extra_objects,
    )

    trajectories_final = torch.cat(
        [trajectories_final_free, trajectories_final_collision], dim=0
    )

    success = compute_success(trajectories_final_free)
    free_fraction = compute_free_fraction(
        trajectories_final_free, trajectories_final_collision
    )
    collision_intensity = (
        compute_collision_intensity(points_final_collision_mask).cpu().numpy()
    )
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

    stats = {
        "task_id": task_id,
        "n_trajectories_per_task": n_trajectories_per_task,
        "success": success,
        "free_fraction": free_fraction,
        "collision_intensity": collision_intensity,
        "path_length_best": path_length_best,
        "sharpness": sharpness,
        "path_length": path_length,
        "waypoints_variance": waypoints_variance,
        "t_task": task_time,
    }

    if not return_full_data:
        return stats

    start_pos = dataset.normalizer.unnormalize(data["start_pos_normalized"])

    goal_pos = dataset.normalizer.unnormalize(data["goal_pos_normalized"])

    full_data = {
        "start_pos": start_pos,
        "goal_pos": goal_pos,
        "trajectories_iters": trajectories_iters,
        "trajectories_final": trajectories_final,
        "trajectories_final_collision": trajectories_final_collision,
        "trajectories_final_free": trajectories_final_free,
        "best_traj_idx": best_traj_idx,
        "traj_final_best": traj_final_best,
    }

    return {"stats": stats, "full_data": full_data}


def get_best_trajectory(
    dataset: TrajectoryDataset,
    data: dict,
    n_trajectories_per_task: int,
    model_wrapper: ModelWrapperBase,
    metric: str = "path-length",  # Options: "path-length", "sharpness"
) -> Dict[str, Any]:
    robot = dataset.robot
    env = dataset.env
    _, trajectories_final = model_wrapper.sample(
        dataset=dataset,
        data=data,
        n_samples=n_trajectories_per_task,
        debug=False,
    )
    if trajectories_final is None:
        return None

    _, trajectories_final_free, _ = env.get_trajectories_collision_and_free(
        trajectories=trajectories_final,
        robot=robot,
        on_extra=model_wrapper.use_extra_objects,
    )

    if trajectories_final_free.shape[0] == 0:
        return None

    if metric == "path-length":
        path_length = compute_path_length(trajectories_final_free, robot)
        best_traj_idx = torch.argmin(path_length).item()
        return trajectories_final_free[best_traj_idx]
    elif metric == "sharpness":
        sharpness = compute_sharpness(trajectories_final_free, robot)
        best_traj_idx = torch.argmin(sharpness).item()
        return trajectories_final_free[best_traj_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run_inference_on_dataset(
    subset: Subset,
    n_tasks: int,
    n_trajectories_per_task: int,
    model_wrapper: ModelWrapperBase,
    debug: bool = False,
) -> Dict[str, Any]:
    statistics = []
    full_data_sample = None
    dataset: TrajectoryDataset = subset.dataset

    return_full_data = True
    cur = 0
    for i in tqdm(range(2 * n_tasks), desc="Processing tasks"):
        idx = np.random.choice(subset.indices)
        data = dataset[idx]

        task_results = run_inference_for_task(
            task_id=i,
            dataset=dataset,
            data=data,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            return_full_data=return_full_data,
            debug=debug,
        )

        if task_results is not None:
            cur += 1
            if return_full_data:
                statistics.append(task_results["stats"])
                full_data_sample = task_results["full_data"]
                return_full_data = False
            else:
                statistics.append(task_results)
        if cur >= n_tasks:
            break

    result = {"statistics": statistics}

    if full_data_sample is not None:
        result["start_pos_sample"] = full_data_sample["start_pos"]
        result["goal_pos_sample"] = full_data_sample["goal_pos"]
        result["trajectories_iters_sample"] = full_data_sample["trajectories_iters"]
        result["trajectories_final_sample"] = full_data_sample["trajectories_final"]
        result["trajectories_final_collision_sample"] = full_data_sample[
            "trajectories_final_collision"
        ]
        result["trajectories_final_free_sample"] = full_data_sample[
            "trajectories_final_free"
        ]
        result["best_traj_idx_sample"] = full_data_sample["best_traj_idx"]
        result["traj_final_best_sample"] = full_data_sample["traj_final_best"]

    return result


def compute_stats(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    statistics = results.get("statistics", [])
    if len(statistics) == 0:
        return None

    n_tasks = len(statistics)
    n_trajectories_per_task = statistics[0]["n_trajectories_per_task"]
    success = [r["success"] for r in statistics]
    free_fraction = [r["free_fraction"] for r in statistics]
    collision_intensity = [r["collision_intensity"] for r in statistics]
    t = [r["t_task"] for r in statistics]
    path_length_best = [
        r["path_length_best"] for r in statistics if r["path_length_best"] is not None
    ]
    sharpness = [
        r["sharpness"].mean().item() for r in statistics if r["sharpness"] is not None
    ]
    path_length = [
        r["path_length"].mean().item()
        for r in statistics
        if r["path_length"] is not None
    ]
    waypoints_variance = [
        r["waypoints_variance"]
        for r in statistics
        if r["waypoints_variance"] is not None
    ]

    time_center, time_hw = bootstrap_confidence_interval(t)
    success_center, success_hw = bootstrap_confidence_interval(success)
    free_fraction_center, free_fraction_hw = bootstrap_confidence_interval(
        free_fraction
    )
    collision_intensity_center, collision_intensity_hw = bootstrap_confidence_interval(
        collision_intensity
    )
    path_length_best_center, path_length_best_hw = (
        bootstrap_confidence_interval(path_length_best)
        if path_length_best
        else (None, None)
    )
    sharpness_center, sharpness_hw = (
        bootstrap_confidence_interval(sharpness) if sharpness else (None, None)
    )
    path_length_center, path_length_hw = (
        bootstrap_confidence_interval(path_length) if path_length else (None, None)
    )
    waypoints_variance_center, waypoints_variance_hw = (
        bootstrap_confidence_interval(waypoints_variance)
        if waypoints_variance
        else (None, None)
    )

    stats = {
        "n_tasks": n_tasks,
        "n_trajectories_per_task": n_trajectories_per_task,
        "time_center": float(time_center),
        "time_hw": float(time_hw),
        "success_rate_center": float(success_center),
        "success_rate_hw": float(success_hw),
        "free_fraction_center": float(free_fraction_center),
        "free_fraction_hw": float(free_fraction_hw),
        "collision_intensity_center": float(collision_intensity_center),
        "collision_intensity_hw": float(collision_intensity_hw),
        "path_length_best_center": float(path_length_best_center) if path_length_best_center is not None else None,
        "path_length_best_hw": path_length_best_hw,
        "sharpness_center": float(sharpness_center) if sharpness_center is not None else None,
        "sharpness_hw": float(sharpness_hw) if sharpness_hw is not None else None,
        "path_length_center": float(path_length_center) if path_length_center is not None else None,
        "path_length_hw": float(path_length_hw) if path_length_hw is not None else None,
        "waypoints_variance_center": float(waypoints_variance_center) if waypoints_variance_center is not None else None,
        "waypoints_variance_hw": float(waypoints_variance_hw) if waypoints_variance_hw is not None else None,
    }
    return stats


def print_stats(results):
    print("=" * 80)

    for split_name in ["train", "val", "test"]:
        stats = results.get(f"{split_name}_stats")
        if stats is None:
            continue
        print(f"-------- {split_name.upper()} SPLIT --------")
        print(f"| n_tasks | {stats['n_tasks']} |")
        print(f"| n_trajectories_per_task | {stats['n_trajectories_per_task']} |")
        print(
            f"| Time to generate n_trajectories_per_task | {stats['time_center']:.3f} ± {stats['time_hw']:.3f} sec |"
        )
        print(
            f"| Success rate | {stats['success_rate_center'] * 100:.2f} ± {stats['success_rate_hw'] * 100:.2f}% |"
        )
        print(
            f"| Free fraction | {stats['free_fraction_center'] * 100:.2f} ± {stats['free_fraction_hw'] * 100:.2f}% |"
        )
        print(
            f"| Collision intensity | {stats['collision_intensity_center'] * 100:.2f} ± {stats['collision_intensity_hw'] * 100:.2f}% |"
        )
        if stats["path_length_best_center"] is not None:
            print(
                f"| Best path length | {stats['path_length_best_center']:.4f} ± {stats['path_length_best_hw']:.4f} |"
            )
        if stats["sharpness_center"] is not None:
            print(
                f"| Sharpness | {stats['sharpness_center']:.4f} ± {stats['sharpness_hw']:.4f} |"
            )
        if stats["path_length_center"] is not None:
            print(
                f"| Path length | {stats['path_length_center']:.4f} ± {stats['path_length_hw']:.4f} |"
            )
        if stats["waypoints_variance_center"] is not None:
            print(
                f"| Waypoints variance | {stats['waypoints_variance_center']:.4f} ± {stats['waypoints_variance_hw']:.4f} |"
            )


def visualize_results(
    results: Dict[str, Any],
    dataset: TrajectoryDataset,
    use_extra_objects: bool,
    results_dir: str,
    generate_animation: bool = True,
    name_prefix: str = "task0",
):
    planner_visualizer = Visualizer(
        env=dataset.env, robot=dataset.robot, use_extra_objects=use_extra_objects
    )
    start_pos = results["start_pos_sample"]
    goal_pos = results["goal_pos_sample"]
    trajectories_iters = results["trajectories_iters_sample"]
    trajectories_final = results["trajectories_final_sample"]
    best_traj_idx = results["best_traj_idx_sample"]

    planner_visualizer.render_scene(
        trajectories=trajectories_final,
        best_traj_idx=best_traj_idx,
        start_pos=start_pos,
        goal_pos=goal_pos,
        save_path=os.path.join(results_dir, f"{name_prefix}-trajectories.png"),
    )

    if not generate_animation:
        return

    planner_visualizer.animate_robot_motion(
        trajectories=trajectories_final,
        best_traj_idx=best_traj_idx,
        start_pos=start_pos,
        goal_pos=goal_pos,
        save_path=os.path.join(results_dir, f"{name_prefix}-robot-motion.mp4"),
        n_frames=min(60, trajectories_final.shape[1]),
    )

    if trajectories_iters is not None:
        planner_visualizer.animate_optimization_iterations(
            trajectories=trajectories_iters,
            best_traj_idx=best_traj_idx,
            start_pos=start_pos,
            goal_pos=goal_pos,
            save_path=os.path.join(results_dir, f"{name_prefix}-opt-iters.mp4"),
            n_frames=min(60, len(trajectories_iters)),
        )


def create_test_subset(
    dataset: TrajectoryDataset,
    n_tasks: int,
    threshold_start_goal_pos: float,
    use_extra_objects: bool = False,
) -> Optional[Subset]:
    start_pos, goal_pos, success = dataset.robot.random_collision_free_start_goal(
        env=dataset.env,
        n_samples=n_tasks,
        threshold_start_goal_pos=threshold_start_goal_pos,
        use_extra_objects=use_extra_objects,
    )
    if not success:
        print(
            "Could not find sufficient collision-free start/goal pairs for test tasks, "
            "try reducing the threshold, robot margin or object density"
        )
        return None

    test_dataset = copy(dataset)
    test_dataset.n_trajs = n_tasks
    test_dataset.start_pos = start_pos
    test_dataset.goal_pos = goal_pos
    test_dataset.start_pos_normalized = test_dataset.normalizer.normalize(start_pos)
    test_dataset.goal_pos_normalized = test_dataset.normalizer.normalize(goal_pos)
    return Subset(test_dataset, list(range(n_tasks)))


def run_inference(
    model_config: ModelConfigBase,
    dataset: TrajectoryDataset,
    train_subset: Optional[Subset],
    val_subset: Optional[Subset],
    test_subset: Optional[Subset],
    results_dir: str,
    n_tasks: int,
    n_trajectories_per_task: int,
    debug: bool,
    tensor_args: Dict[str, Any],
) -> Dict[str, Any]:
    model_wrapper = model_config.prepare(
        dataset=dataset, tensor_args=tensor_args, n_trajectories_per_task=n_trajectories_per_task
    )

    results = {}
    results_json = {}

    print("=" * 80)
    print(f"Starting trajectory generation for {n_tasks} tasks per split")

    if train_subset is not None:
        print("=" * 80)
        print("Processing TRAIN split...")
        results["train"] = run_inference_on_dataset(
            subset=train_subset,
            n_tasks=n_tasks,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            debug=debug,
        )
        results_json["train_stats"] = compute_stats(results["train"])
    if val_subset is not None:
        print("=" * 80)
        print("Processing VAL split...")
        results["val"] = run_inference_on_dataset(
            subset=val_subset,
            n_tasks=n_tasks,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            debug=debug,
        )
        results_json["val_stats"] = compute_stats(results["val"])
    if test_subset is not None:
        print("=" * 80)
        print("Processing TEST split...")
        results["test"] = run_inference_on_dataset(
            subset=test_subset,
            n_tasks=n_tasks,
            n_trajectories_per_task=n_trajectories_per_task,
            model_wrapper=model_wrapper,
            debug=debug,
        )
        results_json["test_stats"] = compute_stats(results["test"])

    print_stats(results)

    if debug:
        print("Saving data...")
        with open(os.path.join(results_dir, "results.pickle"), "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(results_dir, "stats.json"), "w") as f:
            json.dump(results_json, f, indent=4)

        vis_results = None
        if "test" in results and results["test"].get("start_pos_sample") is not None:
            vis_results = results["test"]
        elif "val" in results and results["val"].get("start_pos_sample") is not None:
            vis_results = results["val"]
        elif (
            "train" in results and results["train"].get("start_pos_sample") is not None
        ):
            vis_results = results["train"]

        if vis_results is not None:
            visualize_results(
                results=vis_results,
                dataset=dataset,
                use_extra_objects=model_config.use_extra_objects,
                results_dir=results_dir,
                generate_animation=False,
            )

    return results
