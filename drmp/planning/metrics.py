import numpy as np
import torch
from scipy import stats

from drmp.world.robot import RobotBase


def compute_path_length(trajectories: torch.Tensor, robot: RobotBase):
    assert trajectories.ndim == 3
    if trajectories.shape[0] == 0:
        return torch.tensor(0.0)
    trajectories_pos = robot.get_position(trajectories)
    path_length = torch.linalg.norm(torch.diff(trajectories_pos, dim=-2), dim=-1).sum(
        -1
    )
    return path_length


def compute_sharpness(trajectories: torch.Tensor, robot: RobotBase):
    assert trajectories.ndim == 3
    if trajectories.shape[0] == 0:
        return torch.tensor(0.0)
    trajectories_vel = robot.get_velocity(trajectories, compute=True)
    sharpness = torch.linalg.norm(torch.diff(trajectories_vel, dim=-2), dim=-1).sum(-1)
    return sharpness


def compute_waypoints_variance(trajectories: torch.Tensor, robot: RobotBase):
    assert trajectories.ndim == 3
    if trajectories.shape[0] < 3:
        return torch.tensor(0.0)
    trajectories_pos = (
        robot.get_position(trajectories).permute(1, 0, 2)
        if robot is not None
        else trajectories.permute(1, 0, 2)
    )
    mean = torch.mean(trajectories_pos, dim=-2, keepdim=True)
    cov = (
        (trajectories_pos.unsqueeze(-2) - mean.unsqueeze(-2))
        * (trajectories_pos.unsqueeze(-1) - mean.unsqueeze(-1))
    ).sum(dim=-3) / (trajectories_pos.shape[-2] - 1)
    vol_sq = (cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] ** 2).mean()
    vol = vol_sq.sqrt()
    lin = vol.sqrt()
    return lin


def compute_free_fraction(
    trajectories_free: torch.Tensor, trajectories_collision: torch.Tensor
):
    assert trajectories_free.ndim == 3
    assert trajectories_collision.ndim == 3
    cnt_free = trajectories_free.shape[0]
    cnt_coll = trajectories_collision.shape[0]
    fraction = cnt_free / (cnt_free + cnt_coll)
    return fraction


def compute_collision_intensity(trajectories_collision_mask: torch.Tensor):
    assert trajectories_collision_mask.ndim == 2
    intensity = trajectories_collision_mask.sum() / trajectories_collision_mask.numel()
    return intensity


def compute_success(trajectories_free: torch.Tensor):
    assert trajectories_free.ndim == 3
    return float(trajectories_free.nelement() > 0)


def bootstrap_confidence_interval(
    data: list, confidence_level: float = 0.95, n_resamples: int = 10000
):
    if len(data) in (0, 1):
        return None, None

    res = stats.bootstrap(
        (data,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
    )

    ci_lower = res.confidence_interval.low
    ci_upper = res.confidence_interval.high

    center = (ci_lower + ci_upper) / 2
    half_width = (ci_upper - ci_lower) / 2

    return center, half_width
