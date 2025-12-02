from typing import List
import torch

from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


class FieldFactor:
    def __init__(
        self,
        n_dof: int,
        sigma: float,
        traj_range: List,
        on_extra: bool = False,
    ):
        self.sigma = sigma
        self.n_dof = n_dof
        self.traj_range = traj_range
        self.on_extra = on_extra
        self.K = 1.0 / (sigma**2)

    def get_error(
        self,
        q_trajs: torch.Tensor,
        env: EnvBase,
        robot: Robot,
        calc_jacobian: bool = True,
    ):
        qs = q_trajs[:, self.traj_range[0] : self.traj_range[1], :]
        error = env.compute_cost(qs=qs, robot=robot, on_extra=self.on_extra).reshape(qs.shape[:2])

        if calc_jacobian:
            H = -torch.autograd.grad(error.sum(), q_trajs)[0][
                :, self.traj_range[0] : self.traj_range[1], : self.n_dof
            ]
            error = error.detach()
            return error, H
        else:
            return error
