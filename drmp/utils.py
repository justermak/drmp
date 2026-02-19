import random
from functools import cache
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.interpolate import make_interp_spline


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_config_from_yaml(path: str):
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def save_config_to_yaml(params: dict, path: str):
    with open(path, "w") as outfile:
        yaml.dump(params, outfile, default_flow_style=False)


@cache
def _get_knots(
    n_control_points: int, degree: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    n_internal = n_control_points - degree - 1
    internal_knots = np.linspace(0, 1, n_internal + 2)
    knots = np.concatenate(([0.0] * degree, internal_knots, [1.0] * degree))
    knots_torch = torch.tensor(knots, device=device, dtype=dtype)
    return knots_torch


@cache
def _compute_bspline_basis(
    n_support_points: int,
    n_control_points: int,
    degree: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    knots = _get_knots(n_control_points, degree, device=device, dtype=dtype)
    t = torch.linspace(0, 1, n_support_points, device=device, dtype=dtype)
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

    basis = _compute_bspline_basis(
        n_support_points=n_support_points,
        n_control_points=n_control_points,
        degree=degree,
        device=device,
        dtype=dtype,
    )
    n_fixed = degree - 1

    basis_start = basis[:, :n_fixed]
    basis_mid = basis[:, n_fixed:-n_fixed]
    basis_end = basis[:, -n_fixed:]

    start_pos = trajectories[..., 0, :]
    goal_pos = trajectories[..., -1, :]

    term_start = basis_start.sum(dim=1, keepdim=True) @ start_pos.unsqueeze(-2)
    term_end = basis_end.sum(dim=1, keepdim=True) @ goal_pos.unsqueeze(-2)

    residuals = trajectories - term_start - term_end

    basis_mid_pinv = torch.linalg.pinv(basis_mid)
    control_points_mid = basis_mid_pinv @ residuals
    control_points_start = torch.stack([start_pos] * n_fixed, dim=-2)
    control_points_end = torch.stack([goal_pos] * n_fixed, dim=-2)

    control_points = torch.cat(
        [control_points_start, control_points_mid, control_points_end], dim=-2
    )

    return control_points


def get_trajectories_from_bsplines(
    control_points: torch.Tensor,
    n_support_points: int,
    degree: int,
) -> torch.Tensor:
    device = control_points.device
    dtype = control_points.dtype
    n_control_points = control_points.shape[-2]

    basis = _compute_bspline_basis(
        n_support_points=n_support_points,
        n_control_points=n_control_points,
        degree=degree,
        device=device,
        dtype=dtype,
    )
    trajectories = torch.einsum("sc,...cd->...sd", basis, control_points)

    return trajectories


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
    ).transpose(-2, -1)

    return trajectories_interpolate
