from typing import Any, Dict, List, Tuple

import torch

from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


class UnaryFactor:
    def __init__(
        self,
        dim: int,
        sigma: float,
        mean: torch.Tensor,
        tensor_args: Dict[str, Any] = None,
    ):
        self.dim = dim
        self.sigma = sigma
        self.mean = mean
        self.tensor_args = tensor_args
        self.K = torch.eye(dim, **tensor_args) / sigma**2

    def get_error(
        self, x: torch.Tensor, calc_jacobian: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        error = self.mean - x

        if calc_jacobian:
            H = (
                torch.eye(self.dim, **self.tensor_args)
                .unsqueeze(0)
                .repeat(x.shape[0], 1, 1)
            )
            return error.view(x.shape[0], self.dim, 1), H
        else:
            return error


class FieldFactor:
    def __init__(
        self,
        n_dof: int,
        sigma: float,
        traj_range: List,
        use_extra_obstacles: bool = False,
    ):
        self.sigma = sigma
        self.n_dof = n_dof
        self.traj_range = traj_range
        self.use_extra_obstacles = use_extra_obstacles
        self.K = 1.0 / (sigma**2)

    def get_error(
        self,
        q_trajs: torch.Tensor,
        env: EnvBase,
        robot: Robot,
        calc_jacobian: bool = True,
    ):
        qs = q_trajs[:, self.traj_range[0] : self.traj_range[1], :]
        error = env.compute_cost(qs=qs, robot=robot, on_extra=self.use_extra_obstacles)

        if calc_jacobian:
            H = -torch.autograd.grad(error.sum(), q_trajs)[0][
                :, self.traj_range[0] : self.traj_range[1], : self.n_dof
            ]
            error = error.detach()
            return error, H
        else:
            return error


class GPFactor:
    def __init__(
        self,
        dim: int,
        sigma: float,
        dt: float,
        num_factors: int,
        tensor_args: Dict[str, Any],
        Q_c_inv: torch.Tensor = None,
    ):
        self.dim = dim
        self.dt = dt
        self.tensor_args = tensor_args
        self.state_dim = self.dim * 2
        self.num_factors = num_factors
        self.idx1 = torch.arange(0, self.num_factors, device=tensor_args["device"])
        self.idx2 = torch.arange(1, self.num_factors + 1, device=tensor_args["device"])
        self.phi = self.calc_phi()
        if Q_c_inv is None:
            Q_c_inv = torch.eye(dim, **tensor_args) / sigma**2
        self.Q_c_inv = torch.zeros(num_factors, dim, dim, **tensor_args) + Q_c_inv
        self.Q_inv = self.calc_Q_inv()  # shape: [num_factors, state_dim, state_dim]

        self.H1 = self.phi.unsqueeze(0).repeat(self.num_factors, 1, 1)
        self.H2 = -1.0 * torch.eye(self.state_dim, **self.tensor_args).unsqueeze(
            0
        ).repeat(
            self.num_factors,
            1,
            1,
        )

    def calc_phi(self):
        I = torch.eye(self.dim, **self.tensor_args)
        Z = torch.zeros(self.dim, self.dim, **self.tensor_args)
        phi_u = torch.cat((I, self.dt * I), dim=1)
        phi_l = torch.cat((Z, I), dim=1)
        phi = torch.cat((phi_u, phi_l), dim=0)
        return phi

    def calc_Q_inv(self):
        m1 = 12.0 * (self.dt**-3.0) * self.Q_c_inv
        m2 = -6.0 * (self.dt**-2.0) * self.Q_c_inv
        m3 = 4.0 * (self.dt**-1.0) * self.Q_c_inv

        Q_inv_u = torch.cat((m1, m2), dim=-1)
        Q_inv_l = torch.cat((m2, m3), dim=-1)
        Q_inv = torch.cat((Q_inv_u, Q_inv_l), dim=-2)
        return Q_inv

    def get_error(self, x_traj, calc_jacobian=True):
        state_1 = torch.index_select(x_traj, 1, self.idx1).unsqueeze(-1)
        state_2 = torch.index_select(x_traj, 1, self.idx2).unsqueeze(-1)
        error = state_2 - self.phi @ state_1

        if calc_jacobian:
            H1 = self.H1
            H2 = self.H2
            return error, H1, H2
        else:
            return error
