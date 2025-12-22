from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate


def create_straight_line_trajectory(
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    n_support_points: int,
    dt: float,
    tensor_args: Dict[str, Any],
) -> torch.Tensor:
    """
    Creates a straight-line trajectory from start to goal with average velocity.
    
    Args:
        start_pos: Start position tensor of shape (n_dof,)
        goal_pos: Goal position tensor of shape (n_dof,)
        n_support_points: Number of support points in the trajectory
        dt: Time step between waypoints
        tensor_args: Dictionary with device and dtype
        
    Returns:
        Trajectory tensor of shape (n_support_points, 2*n_dof) with positions and velocities
    """
    assert start_pos.ndim == 1, "start_pos must be of shape (n_dof,)"
    assert goal_pos.ndim == 1, "goal_pos must be of shape (n_dof,)"
    assert start_pos.shape == goal_pos.shape
    
    # Create positions along straight line
    alphas = torch.linspace(0, 1, n_support_points, **tensor_args)
    positions = start_pos.unsqueeze(0) + (goal_pos - start_pos).unsqueeze(0) * alphas.unsqueeze(1)
    
    # Create velocities (average velocity for middle points, zero for start and goal)
    velocities = torch.zeros_like(positions)
    avg_vel = (goal_pos - start_pos) / ((n_support_points - 1) * dt)
    velocities[1:-1, :] = avg_vel
    
    # Concatenate positions and velocities
    trajectory = torch.cat((positions, velocities), dim=-1)
    
    return trajectory


def smoothen_trajectory(
    traj_pos: torch.Tensor,
    n_support_points: int,
    dt: float,
    tensor_args: Dict[str, Any],
) -> torch.Tensor:
    assert traj_pos.ndim == 2, "traj_pos must be of shape (n_points, n_dims)"
    traj_pos = traj_pos.cpu().numpy()
    try:
        pos = interpolate.make_interp_spline(
            np.linspace(0, 1, traj_pos.shape[0]), traj_pos, k=3, bc_type="clamped"
        )(np.linspace(0, 1, n_support_points))
        vel = np.zeros_like(pos)
        avg_vel = (traj_pos[-1] - traj_pos[0]) / ((n_support_points - 1) * dt)
        vel[1:-1, :] = avg_vel
        traj = np.concatenate((pos, vel), axis=-1)
        return torch.tensor(traj, **tensor_args)
    except:
        # Trajectory is too short to interpolate, so add last position again and interpolate
        traj_pos = np.concatenate(
            (
                traj_pos,
                traj_pos[-1:] + np.random.normal(0, 0.01, size=traj_pos[-1:].shape),
            ),
            axis=-2,
        )
        return smoothen_trajectory(
            traj_pos,
            n_support_points=n_support_points,
            dt=dt,
            tensor_args=tensor_args,
        )


def interpolate_trajectories(trajectories: torch.Tensor, n_interpolate: int = 0) -> torch.Tensor:
    assert trajectories.ndim == 3, "trajectories must be of shape (n_trajectories, n_waypoints, n_dims)"
    n_waypoints = trajectories.shape[-2]
    n_total_points = (n_waypoints - 1) * (n_interpolate + 1) + 1
    trajectories_interpolate = F.interpolate(
        trajectories.transpose(-2, -1), size=n_total_points, mode="linear", align_corners=True
    ).transpose(-2, -1)

    return trajectories_interpolate
