import torch

from drmp.world.robot import Robot


def compute_path_length(trajs: torch.Tensor, robot: Robot):
    assert trajs.ndim == 3
    trajs_pos = robot.get_position(trajs)
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length


def compute_smoothness(trajs: torch.Tensor, robot: Robot):
    assert trajs.ndim == 3
    trajs_vel = robot.get_velocity(trajs)
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1).sum(-1)
    return smoothness

def compute_waypoints_variance(trajs: torch.Tensor, robot: Robot):
    assert trajs.ndim == 3
    trajs_pos = robot.get_position(trajs).permute(1, 0, 2) if robot is not None else trajs.permute(1, 0, 2)
    mean = torch.mean(trajs_pos, dim=-2, keepdim=True)
    cov = ((trajs_pos.unsqueeze(-2) - mean.unsqueeze(-2)) * (trajs_pos.unsqueeze(-1) - mean.unsqueeze(-1))).sum(dim=-3) / (trajs_pos.shape[-2] - 1)
    vol_sq = (cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] ** 2).mean()
    vol = vol_sq.sqrt()
    lin = vol.sqrt()
    return lin

def compute_free_fraction(trajs_free: torch.Tensor, trajs_collision: torch.Tensor):
    assert trajs_free.ndim == 3
    assert trajs_collision.ndim == 3
    cnt_free = trajs_free.shape[0]
    cnt_coll = trajs_collision.shape[0]
    fraction = cnt_free / (cnt_free + cnt_coll)
    return fraction

def compute_collision_intensity(trajs_collision_mask: torch.Tensor):
    assert trajs_collision_mask.ndim == 2  # (n_trajs, n_waypoints)
    intensity = trajs_collision_mask.sum() / trajs_collision_mask.numel()
    return intensity

def compute_success(trajs_free: torch.Tensor):
    assert trajs_free.ndim == 3
    return float(trajs_free.nelement() > 0)