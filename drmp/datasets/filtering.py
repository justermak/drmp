from typing import Set

import torch

from drmp.planning.metrics import compute_path_length, compute_sharpness
from drmp.world.robot import Robot


def filter_longest_trajectories(
    trajectories: torch.Tensor,
    robot: Robot,
    task_start_idxs: torch.Tensor,
    portion: float,
) -> Set[int]:
    n_tasks = len(task_start_idxs) - 1
    indices_to_exclude = set()
    
    for task_id in range(n_tasks):
        task_start = task_start_idxs[task_id].item()
        task_end = task_start_idxs[task_id + 1].item()
        task_size = task_end - task_start
        
        if task_size == 0:
            continue
        
        task_trajectories = trajectories[task_start:task_end]
        path_lengths = compute_path_length(task_trajectories, robot)
        n_to_filter = int(portion * task_size)
        
        if n_to_filter > 0:
            _, longest_indices = torch.topk(path_lengths, k=n_to_filter)
            global_indices = [task_start + idx.item() for idx in longest_indices]
            indices_to_exclude.update(global_indices)
    
    return indices_to_exclude


def filter_sharpest_trajectories(
    trajectories: torch.Tensor,
    robot: Robot,
    task_start_idxs: torch.Tensor,
    portion: float,
) -> Set[int]:
    if portion <= 0:
        return set()
    
    n_tasks = len(task_start_idxs) - 1
    indices_to_exclude = set()
    
    for task_id in range(n_tasks):
        task_start = task_start_idxs[task_id].item()
        task_end = task_start_idxs[task_id + 1].item()
        task_size = task_end - task_start
        
        if task_size == 0:
            continue
        
        task_trajectories = trajectories[task_start:task_end]
        sharpnesses = compute_sharpness(task_trajectories, robot)
        n_to_filter = int(portion * task_size) // 2 * 2
        
        if n_to_filter > 0:
            _, sharpest_indices = torch.topk(sharpnesses, k=n_to_filter)
            global_indices = [task_start + idx.item() for idx in sharpest_indices]
            indices_to_exclude.update(global_indices)
    
    return indices_to_exclude


def get_filter_functions():
    return {
        "filter_longest_trajectories": filter_longest_trajectories,
        "filter_sharpest_trajectories": filter_sharpest_trajectories,
    }
