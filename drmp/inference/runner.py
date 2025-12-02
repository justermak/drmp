import os
import pickle
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

from drmp.datasets.dataset import TrajectoryDataset
from drmp.models.guides import GuideTrajectories
from drmp.models.models import PlanningModel
from drmp.planning.metrics import *
from drmp.planning.costs.cost_functions import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
)
from drmp.utils.torch_timer import TimerCUDA
from drmp.utils.yaml import save_config_to_yaml
from drmp.utils.visualizer import Visualizer


def run_inference_for_task(
    task_id: int,
    data_normalized: dict,
    n_samples: int,
    model: PlanningModel,
    dataset: TrajectoryDataset,
    guide: GuideTrajectories,
    start_guide_steps_fraction: float,
    n_guide_steps: int,
    n_diffusion_steps_without_noise: int,
    ddim: bool,
) -> Dict[str, Any]:
    robot = dataset.robot
    env = dataset.env
    
    context = model.build_context(data_normalized)
    hard_conds = model.build_hard_conds(data_normalized)

    with TimerCUDA() as timer_model_sampling:
        trajs_normalized_iters = model.run_inference(
            context,
            hard_conds,
            n_samples=n_samples,
            guide=guide,
            n_guide_steps=n_guide_steps,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            ddim=ddim,
        )
    task_time = timer_model_sampling.elapsed

    trajs_iters = dataset.normalizer.unnormalize(trajs_normalized_iters)
    trajs_final = trajs_iters[-1]
    
    
    trajs_final_coll, trajs_final_free, points_final_collision_mask = env.get_trajs_collision_and_free(robot, trajs_final)

    success = compute_success(trajs_final_free)
    free_fraction = compute_free_fraction(trajs_final_free, trajs_final_coll)
    collision_intensity = compute_collision_intensity(points_final_collision_mask)
    traj_final_best = None
    idx_best_traj = None
    cost_best = None
    cost_smoothness = None
    cost_path_length = None
    cost_all = None
    waypoints_variance = None
    
    if trajs_final_free is not None:
        cost_smoothness = compute_smoothness(trajs_final_free, robot)
        cost_path_length = compute_path_length(trajs_final_free, robot)
        cost_all = cost_path_length + cost_smoothness
        idx_best_traj = torch.argmin(cost_all).item()
        traj_final_best = trajs_final_free[idx_best_traj]
        cost_best = torch.min(cost_all).item()
        waypoints_variance = compute_waypoints_variance(
            trajs_final_free, robot
        ).cpu().numpy()
        cost_smoothness = cost_smoothness.cpu().numpy()
        cost_path_length = cost_path_length.cpu().numpy()
        cost_all = cost_all.cpu().numpy()

    task_results = {
        "task_id": task_id,
        "start_state_pos": robot.get_position(dataset.normalizer.unnormalize(data_normalized["start_states_normalized"])),
        "goal_state_pos": robot.get_position(dataset.normalizer.unnormalize(data_normalized["goal_states_normalized"])),
        "trajs_iters": trajs_iters,
        "trajs_final": trajs_final,
        "trajs_final_coll": trajs_final_coll,
        "trajs_final_free": trajs_final_free,
        "success": success,
        "free_fraction": free_fraction,
        "collision_intensity": collision_intensity,
        "idx_best_traj": idx_best_traj,
        "traj_final_best": traj_final_best,
        "cost_best": cost_best,
        "cost_smoothness": cost_smoothness,
        "cost_path_length": cost_path_length,
        "cost_all": cost_all,
        "waypoints_variance": waypoints_variance,
        "t_task": task_time,
    }
    return task_results


def run_inference_on_dataset(
    subset: Subset,
    n_tasks: int,
    model: PlanningModel,
    guide: GuideTrajectories,
    start_guide_steps_fraction: float,
    n_samples: int,
    n_guide_steps: int,
    n_diffusion_steps_without_noise: int,
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
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_guide_steps=n_guide_steps,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            ddim=ddim,
        )
        results.append(task_results)
    
    return results


def compute_stats(results_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(results_list) == 0:
        return None
    
    n_tasks = len(results_list)
    total_success = sum(r["success"] for r in results_list)
    total_free_fraction = sum(r["free_fraction"] for r in results_list)
    total_collision_intensity = sum(r["collision_intensity"] for r in results_list)
    total_time = sum(r["t_task"] for r in results_list)
    
    # Compute average costs for tasks with collision-free trajectories
    costs_best = [r["cost_best"] for r in results_list if r["cost_best"] is not None]
    costs_smoothness = [r["cost_smoothness"].mean().item() for r in results_list if r["cost_smoothness"] is not None]
    costs_path_length = [r["cost_path_length"].mean().item() for r in results_list if r["cost_path_length"] is not None]
    waypoints_variance = [r["waypoints_variance"] for r in results_list if r["waypoints_variance"] is not None]
    
    stats = {
        "n_tasks": n_tasks,
        "total_time": total_time,
        "avg_time_per_task": total_time / n_tasks,
        "success_rate": total_success / n_tasks,
        "avg_free_fraction": total_free_fraction / n_tasks,
        "avg_collision_intensity": total_collision_intensity / n_tasks,
        "avg_cost_best": np.mean(costs_best) if costs_best else None,
        "std_cost_best": np.std(costs_best) if costs_best else None,
        "avg_cost_smoothness": np.mean(costs_smoothness) if costs_smoothness else None,
        "avg_cost_path_length": np.mean(costs_path_length) if costs_path_length else None,
        "avg_waypoints_variance": np.mean(waypoints_variance) if waypoints_variance else None,
    }
    return stats


def print_stats(results):
    print("=" * 80)
    
    for split_name in ["train", "val", "test"]:
        stats = results.get(f"{split_name}_stats")
        if stats is None:
            continue
        print(f"-------- {split_name.upper()} SPLIT --------")
        print(f"  n_tasks: {stats['n_tasks']}")
        print(f"  total_time: {stats['total_time']:.3f} sec")
        print(f"  avg_time_per_task: {stats['avg_time_per_task']:.3f} sec")
        print(f"  success_rate: {stats['success_rate'] * 100:.2f}%")
        print(f"  avg_free_fraction: {stats['avg_free_fraction'] * 100:.2f}%")
        print(f"  avg_collision_intensity: {stats['avg_collision_intensity'] * 100:.2f}%")
        if stats['avg_cost_best'] is not None:
            print(f"  avg_cost_best: {stats['avg_cost_best']:.4f} ± {stats['std_cost_best']:.4f}")
        if stats['avg_cost_smoothness'] is not None:
            print(f"  avg_cost_smoothness: {stats['avg_cost_smoothness']:.4f}")
        if stats['avg_cost_path_length'] is not None:
            print(f"  avg_cost_path_length: {stats['avg_cost_path_length']:.4f}")
        if stats['avg_waypoints_variance'] is not None:
            print(f"  avg_waypoints_variance: {stats['avg_waypoints_variance']:.4f}")
    
def visualize_results(
    results: Dict[str, List[Dict[str, Any]]],
    dataset: TrajectoryDataset,
    generation_dir: str,
    trajectory_duration: float,
):

    robot = dataset.robot
    task = dataset.task
    planner_visualizer = Visualizer(task=task)

    first_task = results["test"][0] if "test" in results and len(results["test"]) > 0 else (results["val"][0] if "val" in results and len(results["val"]) > 0 else results["train"][0])
    start_state_pos = first_task["start_state_pos"]
    goal_state_pos = first_task["goal_state_pos"]
    trajs_iters = first_task["trajs_iters"]
    trajs_final = first_task["trajs_final"]
    traj_final_best = first_task["traj_final_best"]

    trajs_final_pos = robot.get_position(trajs_final)
    trajs_iters_pos = robot.get_position(trajs_iters)

    planner_visualizer.render_scene(
        trajs=trajs_final_pos,
        traj_best=robot.get_position(traj_final_best) if traj_final_best is not None else None,
        start_state=start_state_pos,
        goal_state=goal_state_pos,
        save_path=os.path.join(
            generation_dir, f"task0-trajectories.png"
        ),
    )

    planner_visualizer.animate_optimization_iterations(
        trajs=trajs_iters_pos,
        traj_best=robot.get_position(traj_final_best) if traj_final_best is not None else None,
        start_state=start_state_pos,
        goal_state=goal_state_pos,
        save_path=os.path.join(
            generation_dir, f"task0-opt-iters.mp4"
        ),
        n_frames=min(50, len(trajs_iters_pos)),
        anim_time=5,
    )

    planner_visualizer.animate_robot_trajectories(
        trajs=trajs_final_pos,
        traj_best=robot.get_position(traj_final_best) if traj_final_best is not None else None,
        start_state=start_state_pos,
        goal_state=goal_state_pos,
        save_path=os.path.join(
            generation_dir, f"task0-robot-motion.mp4"
        ),
        n_frames=min(50, trajs_final_pos.shape[1]),
        anim_time=trajectory_duration,
    )

    plt.show()


def run_inference(
    model: PlanningModel,
    dataset: TrajectoryDataset,
    train_subset: Optional[Subset],
    val_subset: Optional[Subset],
    test_subset: Optional[Subset],
    generations_dir: str,
    experiment_name: str,
    trajectory_duration: float,
    use_extra_objects: bool,
    weight_grad_cost_collision: float,
    weight_grad_cost_smoothness: float,
    num_interpolated_points_for_collision: int,
    start_guide_steps_fraction: float,
    n_tasks: int,
    n_samples: int,
    threshold_start_goal_pos: float,
    n_guide_steps: int,
    n_diffusion_steps_without_noise: int,
    ddim: bool,
    debug: bool,
    tensor_args: Dict[str, Any],
) -> Dict[str, Any]:
    
    generation_dir = os.path.join(generations_dir, experiment_name)
    os.makedirs(generation_dir, exist_ok=True)
    robot: Robot = dataset.robot
    
    dt: float = trajectory_duration / (dataset.n_support_points - 1)
    robot.dt = dt
    
    config = {
        "trajectory_duration": trajectory_duration,
        "use_extra_objects": use_extra_objects,
        "weight_grad_cost_collision": weight_grad_cost_collision,
        "weight_grad_cost_smoothness": weight_grad_cost_smoothness,
        "num_interpolated_points_for_collision": num_interpolated_points_for_collision,
        "start_guide_steps_fraction": start_guide_steps_fraction,
        "n_tasks": n_tasks,
        "n_samples": n_samples,
        "threshold_start_goal_pos": threshold_start_goal_pos,
        "n_guide_steps": n_guide_steps,
        "n_diffusion_steps_without_noise": n_diffusion_steps_without_noise,
        "ddim": ddim,
    }
    
    save_config_to_yaml(config, os.path.join(generation_dir, "config.yaml"))


    collision_costs = [
        CostCollision(
            robot = robot,
            env=dataset.env,
            n_support_points=dataset.n_support_points,
            sigma_coll=1.0,
            on_extra=use_extra_objects,
            tensor_args=tensor_args,
        )
    ]
    costs_weights = [weight_grad_cost_collision] * len(collision_costs)

    smoothness_costs = [
        CostGPTrajectory(
            robot, dataset.n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args
        )
    ]
    smoothness_weights = [weight_grad_cost_smoothness]

    costs = collision_costs + smoothness_costs
    weights = costs_weights + smoothness_weights

    cost_composite = CostComposite(
        robot,
        dataset.n_support_points,
        costs=costs,
        weights=weights,
        tensor_args=tensor_args,
    )

    guide = GuideTrajectories(
        dataset,
        cost_composite,
        do_clip_grad=True,
        interpolate_trajectories_for_collision=True,
        num_interpolated_points_for_collision=num_interpolated_points_for_collision
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
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_samples=n_samples,
            n_guide_steps=n_guide_steps,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            ddim=ddim,
        )
        results["train_stats"] = compute_stats(results["train"])
    if val_subset is not None:
        results["val"] = run_inference_on_dataset(
            subset=val_subset,
            n_tasks=n_tasks,
            model=model,
            guide=guide,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_samples=n_samples,
            n_guide_steps=n_guide_steps,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            ddim=ddim,
        )
        results["val_stats"] = compute_stats(results["val"])
    if test_subset is not None:
        results["test"] = run_inference_on_dataset(
            subset=test_subset,
            n_tasks=n_tasks,
            model=model,
            guide=guide,
            start_guide_steps_fraction=start_guide_steps_fraction,
            n_samples=n_samples,
            n_guide_steps=n_guide_steps,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            ddim=ddim,
        )
        results["test_stats"] = compute_stats(results["test"])


    print_stats(results)

    with open(os.path.join(generation_dir, "results_data_dict.pickle"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    if debug:
        visualize_results(results=results, dataset=dataset, generation_dir=generation_dir, trajectory_duration=trajectory_duration)
    
    return results