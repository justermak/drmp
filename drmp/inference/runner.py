import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Subset

from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference.guides import GuideTrajectories
from drmp.models.diffusion import PlanningModel
from drmp.world.robot import Robot
from drmp.planning.costs.cost_functions import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
)
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
    data_normalized: dict,
    n_samples: int,
    model: PlanningModel,
    dataset: TrajectoryDataset,
    guide: GuideTrajectories,
    use_extra_objects: bool,
    start_guide_steps_fraction: float,
    n_guide_steps: int,
    ddim: bool,
) -> Dict[str, Any]:
    robot = dataset.robot
    env = dataset.env

    context = model.build_context(data_normalized)
    hard_conds = model.build_hard_conditions(data_normalized)

    with TimerCUDA() as timer_model_sampling:
        trajectories_normalized_iters = model.run_inference(
            context,
            hard_conds,
            n_samples=n_samples,
            guide=guide,
            n_guide_steps=n_guide_steps,
            start_guide_steps_fraction=start_guide_steps_fraction,
            ddim=ddim,
        )
    task_time = timer_model_sampling.elapsed

    trajectories_iters = dataset.normalizer.unnormalize(trajectories_normalized_iters)
    trajectories_final = trajectories_iters[-1]

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
            dataset.normalizer.unnormalize(data_normalized["start_states_normalized"])
        ),
        "goal_pos": robot.get_position(
            dataset.normalizer.unnormalize(data_normalized["goal_states_normalized"])
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
    model: PlanningModel,
    guide: GuideTrajectories,
    use_extra_objects: bool,
    start_guide_steps_fraction: float,
    n_samples: int,
    n_guide_steps: int,
    ddim: bool,
) -> List[Dict[str, Any]]:
    results = []
    dataset: TrajectoryDataset = subset.dataset

    for i in range(n_tasks):
        idx = np.random.choice(subset.indices)
        data_normalized = dataset[idx]

        task_results = run_inference_for_task(
            task_id=i,
            data_normalized=data_normalized,
            n_samples=n_samples,
            model=model,
            dataset=dataset,
            guide=guide,
            use_extra_objects=use_extra_objects,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_guide_steps=n_guide_steps,
            ddim=ddim,
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
    time = [r["t_task"] for r in results]
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
        "time_avg": np.mean(time),
        "time_std": np.std(time),
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
            print(f"sharpness: {stats['sharpness_avg']:.4f}")
        if stats["path_length_avg"] is not None:
            print(f"Path length: {stats['path_length_avg']:.4f}")
        if stats["waypoints_variance_avg"] is not None:
            print(f"Waypoints variance: {stats['waypoints_variance_avg']:.4f}")


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
        save_path=os.path.join(generation_dir, f"task0-trajectories.png"),
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

def run_inference(
    model: PlanningModel,
    dataset: TrajectoryDataset,
    train_subset: Optional[Subset],
    val_subset: Optional[Subset],
    test_subset: Optional[Subset],
    generations_dir: str,
    experiment_name: str,
    use_extra_objects: bool,
    sigma_collision: float,
    sigma_gp: float,
    do_clip_grad: bool,
    max_grad_norm: float,
    n_interpolate: int,
    start_guide_steps_fraction: float,
    n_tasks: int,
    n_samples: int,
    threshold_start_goal_pos: float,
    n_guide_steps: int,
    ddim: bool,
    debug: bool,
    tensor_args: Dict[str, Any],
) -> Dict[str, Any]:
    generation_dir = os.path.join(generations_dir, experiment_name)
    os.makedirs(generation_dir, exist_ok=True)
    robot: Robot = dataset.robot
    
    config = {
        "use_extra_objects": use_extra_objects,
        "start_guide_steps_fraction": start_guide_steps_fraction,
        "sigma_collision": sigma_collision,
        "sigma_gp": sigma_gp,
        "do_clip_grad": do_clip_grad,
        "max_grad_norm": max_grad_norm,
        "n_interpolate": n_interpolate,
        "n_tasks": n_tasks,
        "n_samples": n_samples,
        "threshold_start_goal_pos": threshold_start_goal_pos,
        "n_guide_steps": n_guide_steps,
        "ddim": ddim,
    }

    save_config_to_yaml(config, os.path.join(generation_dir, "config.yaml"))

    collision_costs = [
        CostCollision(
            robot=robot,
            env=dataset.env,
            n_support_points=dataset.n_support_points,
            sigma_collision=sigma_collision,
            use_extra_obstacles=use_extra_objects,
            tensor_args=tensor_args,
        )
    ]

    sharpness_costs = [
        CostGPTrajectory(
            robot=robot, 
            n_support_points=dataset.n_support_points, 
            sigma_gp=sigma_gp, 
            tensor_args=tensor_args
        )
    ]

    costs = collision_costs + sharpness_costs

    cost = CostComposite(
        robot=robot,
        n_support_points=dataset.n_support_points,
        costs=costs,
        tensor_args=tensor_args,
    )

    guide = GuideTrajectories(
        dataset=dataset,
        cost=cost,
        do_clip_grad=do_clip_grad,
        max_grad_norm=max_grad_norm,
        n_interpolate=n_interpolate,
    )

    results = {}

    print(f"{'=' * 80}")
    print(f"Starting trajectory generation for {n_tasks} tasks per split")

    if train_subset is not None:
        results["train"] = run_inference_on_dataset(
            subset=train_subset,
            n_tasks=n_tasks,
            model=model,
            guide=guide,
            use_extra_objects=use_extra_objects,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_samples=n_samples,
            n_guide_steps=n_guide_steps,
            ddim=ddim,
        )
        results["train_stats"] = compute_stats(results["train"])
    if val_subset is not None:
        results["val"] = run_inference_on_dataset(
            subset=val_subset,
            n_tasks=n_tasks,
            model=model,
            guide=guide,
            use_extra_objects=use_extra_objects,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_samples=n_samples,
            n_guide_steps=n_guide_steps,
            ddim=ddim,
        )
        results["val_stats"] = compute_stats(results["val"])
    if test_subset is not None:
        results["test"] = run_inference_on_dataset(
            subset=test_subset,
            n_tasks=n_tasks,
            model=model,
            guide=guide,
            use_extra_objects=use_extra_objects,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_samples=n_samples,
            n_guide_steps=n_guide_steps,
            ddim=ddim,
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
