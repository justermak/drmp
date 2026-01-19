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
    alphas = torch.linspace(0, 1, n_support_points, **tensor_args)
    positions = start_pos.unsqueeze(0) + (goal_pos - start_pos).unsqueeze(
        0
    ) * alphas.unsqueeze(1)
    velocities = torch.zeros_like(positions)
    avg_vel = (goal_pos - start_pos) / ((n_support_points - 1) * dt)
    velocities[1:-1, :] = avg_vel
    trajectory = torch.cat((positions, velocities), dim=-1)

    return trajectory


def _get_knots(n_control_points: int, degree: int) -> np.ndarray:
    n_internal = n_control_points - degree - 1
    internal_knots = np.linspace(0, 1, n_internal + 2)
    knots = np.concatenate(
        ([0.0] * degree, internal_knots, [1.0] * degree)
    )
    return knots


def _compute_bspline_basis(
    t: torch.Tensor, knots: torch.Tensor, degree: int, n_control_points: int
) -> torch.Tensor:
    n_t = t.shape[0]
    device = t.device
    dtype = t.dtype

    num_basis_0 = n_control_points + degree
    basis = torch.zeros(n_t, num_basis_0, device=device, dtype=dtype)

    for i in range(num_basis_0):
        cond = (t >= knots[i]) & (t < knots[i + 1])
        basis[:, i] = cond.type(dtype)

    for d in range(1, degree + 1):
        num_basis_d = num_basis_0 - d
        new_basis = torch.zeros(n_t, num_basis_d, device=device, dtype=dtype)

        for i in range(num_basis_d):
            denom1 = knots[i + d] - knots[i]
            if denom1 != 0:
                term1 = ((t - knots[i]) / denom1) * basis[:, i]
            else:
                term1 = torch.zeros_like(basis[:, i])

            denom2 = knots[i + d + 1] - knots[i + 1]
            if denom2 != 0:
                term2 = ((knots[i + d + 1] - t) / denom2) * basis[:, i + 1]
            else:
                term2 = torch.zeros_like(basis[:, i + 1])

            new_basis[:, i] = term1 + term2
        basis = new_basis

    if t.numel() > 0:
        t_max = knots[-1]
        mask_end = t == t_max
        if mask_end.any():
            basis[mask_end, :] = 0
            basis[mask_end, -1] = 1.0

    return basis


def fit_bsplines_to_trajectories(
    trajectories: torch.Tensor,
    n_control_points: int,
    degree: int,
) -> torch.Tensor:
    device = trajectories.device
    dtype = trajectories.dtype
    n_support_points = trajectories.shape[-2]

    knots_np = _get_knots(n_control_points, degree)
    knots = torch.tensor(knots_np, device=device, dtype=dtype)
    t_eval = torch.linspace(0, 1, n_support_points, device=device, dtype=dtype)
    basis = _compute_bspline_basis(t_eval, knots, degree, n_control_points)
    basis_pinv = torch.linalg.pinv(basis)
    control_points = torch.matmul(basis_pinv, trajectories)

    return control_points


def get_trajectories_from_bsplines(
    control_points: torch.Tensor,
    n_support_points: int,
    degree: int,
) -> torch.Tensor:
    device = control_points.device
    dtype = control_points.dtype
    n_control_points = control_points.shape[-2]
    
    knots_np = _get_knots(n_control_points, degree)
    knots = torch.tensor(knots_np, device=device, dtype=dtype)
    t_eval = torch.linspace(0, 1, n_support_points, device=device, dtype=dtype)
    basis = _compute_bspline_basis(t_eval, knots, degree, n_control_points)
    trajectories = torch.einsum("sc,...cd->...sd", basis, control_points)

    return trajectories


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


def interpolate_trajectories(
    trajectories: torch.Tensor, n_interpolate: int = 0
) -> torch.Tensor:
    assert trajectories.ndim == 3, (
        "trajectories must be of shape (n_trajectories, n_waypoints, n_dims)"
    )
    n_waypoints = trajectories.shape[-2]
    n_total_points = (n_waypoints - 1) * (n_interpolate + 1) + 1
    trajectories_interpolate = F.interpolate(
        trajectories.transpose(-2, -1),
        size=n_total_points,
        mode="linear",
        align_corners=True,
    ).transpose(-2, -1)

    return trajectories_interpolate
