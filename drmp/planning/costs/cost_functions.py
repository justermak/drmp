from abc import ABC, abstractmethod
from typing import Any, Dict, List

import einops
import torch

from drmp.config import N_DIM
from drmp.planning.costs.factors import FieldFactor, GPFactor, UnaryFactor
from drmp.utils.trajectory_utils import interpolate_trajectories
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


class Cost(ABC):
    def __init__(
        self,
        robot: Robot,
        n_support_points: int,
        tensor_args: Dict[str, Any],
    ):
        self.robot = robot
        self.n_dof = N_DIM
        self.dim = 2 * N_DIM
        self.n_support_points = n_support_points

        self.tensor_args = tensor_args

    def __call__(self, trajectories, **kwargs):
        return self.eval(trajectories, **kwargs)

    @abstractmethod
    def eval(self, trajectories, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        pass


class CostComposite(Cost):
    def __init__(
        self,
        robot: Robot,
        n_support_points: int,
        costs: List[Cost],
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.costs = costs

    def eval(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        trajectories_interpolated = (
            interpolate_trajectories(trajectories, n_interpolate=n_interpolate)
        )
        cost = 0
        for cost_class in self.costs:
            cost += cost_class(
                trajectories_interpolated, n_interpolate=n_interpolate
            ) if isinstance(cost_class, CostCollision) else cost_class(
                trajectories
            )

        return cost

    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        trajectories.requires_grad = True

        batch_size = trajectories.shape[0]
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost in self.costs:
            A, b, K = cost.get_linear_system(
                trajectories=trajectories, n_interpolate=n_interpolate
            )
            if A is None or b is None or K is None:
                continue
            optim_dim += A.shape[1]
            As.append(A.detach())
            bs.append(b.detach())
            Ks.append(K.detach())

        A = torch.cat(As, dim=1)
        b = torch.cat(bs, dim=1)
        K = torch.zeros(batch_size, optim_dim, optim_dim, **self.tensor_args)
        offset = 0
        for i in range(len(Ks)):
            dim = Ks[i].shape[1]
            K[:, offset : offset + dim, offset : offset + dim] = Ks[i]
            offset += dim
        return A, b, K


class CostCollision(Cost):
    def __init__(
        self,
        robot: Robot,
        env: EnvBase,
        n_support_points: int,
        sigma_collision: float,
        tensor_args: Dict[str, Any],
        use_extra_objects: bool = False,
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.env = env
        self.sigma_collision = sigma_collision
        self.use_extra_objects = use_extra_objects
        self.obst_factor = FieldFactor(
            n_dof=self.n_dof,
            sigma=self.sigma_collision,
            traj_range=[
                1,
                None,
            ],
            use_extra_objects=self.use_extra_objects,
        )

    def eval(self, trajectories: torch.Tensor, n_interpolate: int):
        err_obst = self.obst_factor.get_error(
            q_trajectories=trajectories,
            env=self.env,
            robot=self.robot,
            n_interpolate=n_interpolate,
            calc_jacobian=False,
        )
        w_mat = self.obst_factor.K
        obst_costs = w_mat * err_obst.sum(1)
        costs = obst_costs

        return costs

    def get_linear_system(
        self,
        trajectories: torch.Tensor,
        n_interpolate: int,
    ):
        A, b, K = None, None, None
        batch_size = trajectories.shape[0]

        err_obst, H_obst = self.obst_factor.get_error(
            q_trajectories=trajectories,
            env=self.env,
            robot=self.robot,
            n_interpolate=n_interpolate,
            calc_jacobian=True,
        )

        A = torch.zeros(
            batch_size,
            self.n_support_points - 1,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )
        A[:, :, : H_obst.shape[-1]] = H_obst
        # shift each row by self.dim
        idxs = torch.arange(A.shape[-1], **self.tensor_args).repeat(A.shape[-2], 1)
        idxs = (
            idxs
            - torch.arange(
                self.dim,
                (idxs.shape[0] + 1) * self.dim,
                self.dim,
                **self.tensor_args,
            ).view(-1, 1)
        ) % idxs.shape[-1]
        idxs = idxs.to(torch.int64)
        A = torch.gather(A, -1, idxs.repeat(batch_size, 1, 1))

        # old code not vectorized
        # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L275

        b = err_obst.unsqueeze(-1)
        K = self.obst_factor.K * torch.eye(
            (self.n_support_points - 1), **self.tensor_args
        ).repeat(batch_size, 1, 1)

        return A, b, K


class CostGPTrajectory(Cost):
    def __init__(
        self,
        robot: Robot,
        n_support_points: int,
        sigma_gp: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)

        self.sigma_gp = sigma_gp

        self.gp_prior = GPFactor(
            dim=self.n_dof,
            sigma=self.sigma_gp,
            dt=self.robot.dt,
            num_factors=self.n_support_points - 1,
            tensor_args=self.tensor_args,
        )

    def eval(self, trajectories: torch.Tensor):
        err_gp = self.gp_prior.get_error(trajectories, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs

    def get_linear_system(self, trajectories: torch.Tensor):
        pass


class CostGP(Cost):
    def __init__(
        self,
        robot: Robot,
        n_support_points: int,
        start_state: torch.Tensor,
        sigma_start: float,
        sigma_gp: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.start_state = start_state
        self.sigma_start = sigma_start
        self.sigma_gp = sigma_gp

        self.start_prior = UnaryFactor(
            dim=self.dim,
            sigma=self.sigma_start,
            mean=self.start_state,
            tensor_args=self.tensor_args,
        )

        self.gp_prior = GPFactor(
            dim=self.n_dof,
            sigma=self.sigma_gp,
            dt=self.robot.dt,
            num_factors=self.n_support_points - 1,
            tensor_args=self.tensor_args,
        )

    def eval(self, trajectories: torch.Tensor):
        pass

    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):
        batch_size = trajectories.shape[0]
        A = torch.zeros(
            batch_size,
            self.dim * self.n_support_points,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )
        b = torch.zeros(
            batch_size, self.dim * self.n_support_points, 1, **self.tensor_args
        )
        K = torch.zeros(
            batch_size,
            self.dim * self.n_support_points,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )

        err_p, H_p = self.start_prior.get_error(trajectories[:, [0]])
        A[:, : self.dim, : self.dim] = H_p
        b[:, : self.dim] = err_p
        K[:, : self.dim, : self.dim] = self.start_prior.K

        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajectories)

        A[:, self.dim :, : -self.dim] = torch.block_diag(*H1_gp)
        A[:, self.dim :, self.dim :] += torch.block_diag(*H2_gp)
        b[:, self.dim :] = einops.rearrange(err_gp, "b h d 1 -> b (h d) 1")
        K[:, self.dim :, self.dim :] += torch.block_diag(*self.gp_prior.Q_inv)

        return A, b, K


class CostGoalPrior(Cost):
    def __init__(
        self,
        robot: Robot,
        n_support_points: int,
        goal_state: torch.Tensor,
        n_trajectories: int,
        num_samples: int,
        sigma_goal_prior: float,
        tensor_args: Dict[str, Any],
    ):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.goal_state = goal_state
        self.n_trajectories = n_trajectories
        self.num_samples = num_samples
        self.sigma_goal_prior = sigma_goal_prior

        self.goal_prior = UnaryFactor(
            dim=self.dim,
            sigma=self.sigma_goal_prior,
            mean=self.goal_state,
            tensor_args=self.tensor_args,
        )

    def eval(self, trajectories: torch.Tensor):
        pass

    def get_linear_system(self, trajectories: torch.Tensor, n_interpolate: int):       
        A = torch.zeros(
            self.n_trajectories,
            self.dim,
            self.dim * self.n_support_points,
            **self.tensor_args,
        )

        err_g, H_g = self.goal_prior.get_error(trajectories[:, [-1]])
        A[:, :, -self.dim :] = H_g
        b = err_g
        K = self.goal_prior.K

        return A, b, K
