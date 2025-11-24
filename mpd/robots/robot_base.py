import abc
from abc import ABC

import torch

from mpd.utils.torch_utils import to_numpy, to_torch
from mpd.trajectory.utils import finite_difference_vector


class RobotBase(ABC):
    def __init__(
        self,
        name="RobotBase",
        q_limits=None,
        link_margins_for_object_collision_checking=None,
        link_idxs_for_object_collision_checking=None,
        num_interpolated_points_for_object_collision_checking=1,
        dt=1.0,
        tensor_args=None,
        **kwargs,
    ):
        self.name = name
        self.tensor_args = tensor_args
        self.dt = dt

        # Configuration space
        assert q_limits is not None, "q_limits cannot be None"
        self.q_limits = q_limits
        self.q_min = q_limits[0]
        self.q_max = q_limits[1]
        self.q_min_np = to_numpy(self.q_min)
        self.q_max_np = to_numpy(self.q_max)
        self.q_distribution = torch.distributions.uniform.Uniform(
            self.q_min, self.q_max
        )
        self.q_dim = len(self.q_min)

        # Object collision parameters
        self.num_interpolated_points_for_object_collision_checking = (
            num_interpolated_points_for_object_collision_checking
        )
        self.link_margins_for_object_collision_checking = (
            link_margins_for_object_collision_checking
        )
        self.link_margins_for_object_collision_checking_tensor = to_torch(
            link_margins_for_object_collision_checking, **self.tensor_args
        ).repeat_interleave(num_interpolated_points_for_object_collision_checking)
        self.link_idxs_for_object_collision_checking = (
            link_idxs_for_object_collision_checking
        )

    def fk_map_collision(self, q, **kwargs):
        if q.ndim == 1:
            q = q.unsqueeze(0)  # add batch dimension
        return self.fk_map_collision_impl(q, **kwargs)

    @abc.abstractmethod
    def fk_map_collision_impl(self, q, **kwargs):
        # q: (..., q_dim)
        # return: (..., links_collision_positions, 3)
        raise NotImplementedError

    def get_position(self, x):
        return x[..., : self.q_dim]

    def get_velocity(self, x):
        vel = x[..., self.q_dim : 2 * self.q_dim]
        # If there is no velocity in the state, then compute it via finite difference
        if x.nelement() != 0 and vel.nelement() == 0:
            vel = finite_difference_vector(x, dt=self.dt, method="central")
            return vel
        return vel

    def get_acceleration(self, x):
        acc = x[..., 2 * self.q_dim : 3 * self.q_dim]
        # If there is no acceleration in the state, then compute it via finite difference
        if x.nelement() != 0 and acc.nelement() == 0:
            vel = self.get_velocity(x)
            acc = finite_difference_vector(vel, dt=self.dt, method="central")
            return acc
        return acc

    def random_q(self, n_samples=10):
        q_pos = self.q_distribution.sample((n_samples,))
        return q_pos

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectories(self, ax, trajs=None, **kwargs):
        raise NotImplementedError
