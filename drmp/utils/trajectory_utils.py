from typing import Any, Dict, Callable, List
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate

from drmp.config import DEFAULT_TENSOR_ARGS
from drmp.utils.torch_utils import to_numpy, to_torch


def smoothen_trajectory(
    traj_pos: torch.Tensor,
    n_support_points: int,
    dt: float,
    tensor_args=DEFAULT_TENSOR_ARGS,
) -> torch.Tensor:
    assert traj_pos.ndim == 2, "traj_pos must be of shape (n_points, n_dims)"
    traj_pos = to_numpy(traj_pos)
    try:
        pos = interpolate.make_interp_spline(
            np.linspace(0, 1, traj_pos.shape[0]), traj_pos, k=3, bc_type="clamped"
        )(np.linspace(0, 1, n_support_points))
        vel = np.zeros_like(pos)
        avg_vel = (traj_pos[-1] - traj_pos[0]) / ((n_support_points - 1) * dt)
        vel[1:-1, :] = avg_vel
        traj = np.concatenate((pos, vel), axis=-1)
        return to_torch(traj, **tensor_args)
    except:
        # Trajectory is too short to interpolate, so add last position again and interpolate
        traj_pos = np.concatenate((traj_pos, traj_pos[-1:] + np.random.normal(0, 0.01, size=traj_pos[-1:].shape)), axis=-2)
        return smoothen_trajectory(
            traj_pos,
            n_support_points=n_support_points,
            dt=dt,
            tensor_args=tensor_args,
        )

def interpolate_trajs(trajs: torch.Tensor, n_interp: int = 0) -> torch.Tensor:
    assert trajs.ndim == 3, "trajs must be of shape (n_trajs, n_waypoints, n_dims)"
    n_waypoints = trajs.shape[-2]
    n_total_points = (n_waypoints - 1) * (n_interp + 1) + 1
    trajs_interp = F.interpolate(
        trajs.transpose(-2, -1),
        size=n_total_points,
        mode='linear',
        align_corners=True
    ).transpose(-2, -1)
    
    return trajs_interp


def finite_difference_vector(x, dt: float) -> torch.Tensor:
    # finite differences with zero padding at the borders
    diff_vector = torch.zeros_like(x)
    diff_vector[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2 * dt)
    return diff_vector


def extend_path(distance_fn: Callable, q1: torch.Tensor, q2: torch.Tensor, max_step: float =0.03, max_dist: float =0.1, tensor_args: Dict[str, Any]=DEFAULT_TENSOR_ARGS) -> torch.Tensor:
    # max_dist must be <= radius of RRT star!
    dist = distance_fn(q1, q2)
    if dist > max_dist:
        q2 = q1 + (q2 - q1) * (max_dist / dist)

    alpha = torch.linspace(0, 1, int(dist / max_step) + 2).to(
        **tensor_args
    )  # skip first and last
    q1 = q1.unsqueeze(0)
    q2 = q2.unsqueeze(0)
    extension = q1 + (q2 - q1) * alpha.unsqueeze(1)
    return extension


def safe_path(sequence: torch.Tensor, collision_fn: Callable) -> torch.Tensor:
    in_collision = collision_fn(sequence)
    idxs_in_collision = torch.argwhere(in_collision)
    if idxs_in_collision.nelement() == 0:
        if sequence.ndim == 1:
            return sequence.reshape((1, -1))
        return sequence[-1].reshape((1, -1))
    else:
        first_idx_in_collision = idxs_in_collision[0]
        if first_idx_in_collision == 0:
            # the first point in the sequence is in collision
            return torch.empty((0, sequence.shape[-1]), device=sequence.device)
        # return the point immediate before the one in collision
        return sequence[first_idx_in_collision - 1]


def purge_duplicates_from_traj(path, eps=1e-6):
    # Remove duplicated points from a trajectory
    if len(path) < 2:
        return path
    if isinstance(path, list):
        path = torch.stack(path, dim=0)
    if path.shape[0] == 2:
        return path

    abs_diff = torch.abs(torch.diff(path, dim=-2))
    row_idxs = torch.argwhere(
        torch.all((abs_diff > eps) == False, dim=-1) == False
    ).unique()
    selection = path[row_idxs]
    # Always add the first and last elements of the path
    if torch.allclose(selection[0], path[0]) is False:
        selection = torch.cat((path[0].view(1, -1), selection), dim=0)
    if torch.allclose(selection[-1], path[-1]) is False:
        selection = torch.cat((selection, path[-1].view(1, -1)), dim=0)
    return selection
