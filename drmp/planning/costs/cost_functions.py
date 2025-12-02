from abc import ABC, abstractmethod
from typing import Any, Dict, List

import einops
import torch

from drmp.planning.costs.field_factor import FieldFactor
from drmp.planning.costs.gp_factor import GPFactor
from drmp.planning.costs.unary_factor import UnaryFactor
from drmp.config import DEFAULT_TENSOR_ARGS, N_DIMS
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


class Cost(ABC):
    def __init__(self, robot: Robot, n_support_points: int, tensor_args: Dict[str, Any]=DEFAULT_TENSOR_ARGS):
        self.robot = robot
        self.n_dof = N_DIMS
        self.dim = 2 * N_DIMS
        self.n_support_points = n_support_points

        self.tensor_args = tensor_args

    def set_cost_factors(self):
        pass

    def __call__(self, trajs, **kwargs):
        return self.eval(trajs, **kwargs)

    @abstractmethod
    def eval(self, trajs, **kwargs):
        pass

    @abstractmethod
    def get_linear_system(self, trajs: torch.Tensor):
        pass

    def get_q_pos_vel_and_fk_map(self, trajs: torch.Tensor):
        q_pos = self.robot.get_position(trajs)
        q_vel = self.robot.get_velocity(trajs)
        return trajs, q_pos, q_vel


class CostComposite(Cost):
    def __init__(self, robot: Robot, n_support_points: int, costs: List[Cost], weights: List[float]=None, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.costs = costs
        self.weights = weights or [1.0] * len(costs)

    def eval(
        self,
        trajs: torch.Tensor,
        trajs_interpolated: torch.Tensor = None,
        return_invidual_costs_and_weights: bool = False,
        **kwargs,
    ):
        q_pos = self.robot.get_position(trajs)
        q_vel = self.robot.get_velocity(trajs)

        if not return_invidual_costs_and_weights:
            cost_total = 0
            for cost, weight in zip(self.costs, self.weights):
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    if isinstance(cost, CostCollision):
                        trajs_tmp = trajs_interpolated
                    else:
                        trajs_tmp = trajs
                else:
                    trajs_tmp = trajs
                cost_tmp = weight * cost(
                    trajs_tmp,
                    q_pos=q_pos,
                    q_vel=q_vel,
                    **kwargs,
                )
                cost_total += cost_tmp
            return cost_total
        else:
            cost_l = []
            for cost in self.costs:
                if trajs_interpolated is not None:
                    # Compute only collision costs with interpolated trajectories.
                    # Other costs are computed with non-interpolated trajectories, e.g. smoothness
                    if isinstance(cost, CostCollision):
                        trajs_tmp = trajs_interpolated
                    else:
                        trajs_tmp = trajs
                else:
                    trajs_tmp = trajs

                cost_tmp = cost(
                    trajs_tmp,
                    q_pos=q_pos,
                    q_vel=q_vel,
                    **kwargs,
                )
                cost_l.append(cost_tmp)

            if return_invidual_costs_and_weights:
                return cost_l, self.weights

    def get_linear_system(self, trajs: torch.Tensor):
        trajs.requires_grad = True

        batch_size = trajs.shape[0]
        As, bs, Ks = [], [], []
        optim_dim = 0
        for cost, weight_cost in zip(self.costs, self.weights):
            A, b, K = cost.get_linear_system(
                trajs,
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
    def __init__(self, robot: Robot,  env: EnvBase, n_support_points: int, sigma_coll: float=None, on_extra: bool=False, tensor_args: Dict[str, Any]=DEFAULT_TENSOR_ARGS):
        super().__init__(robot, n_support_points, tensor_args=tensor_args)
        self.env = env
        self.sigma_coll = sigma_coll
        self.on_extra = on_extra
        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.obst_factor = FieldFactor(
            n_dof=self.n_dof,
            sigma=self.sigma_coll,
            traj_range=[1, None],  # take the whole trajectory except for the first point
            on_extra=self.on_extra,
        )

    def eval(self, trajs: torch.Tensor):
        err_obst = self.obst_factor.get_error(
            q_trajs=trajs,
            env=self.env,
            robot=self.robot,
            calc_jacobian=False,
        )
        w_mat = self.obst_factor.K
        obst_costs = w_mat * err_obst.sum(1)
        costs = obst_costs

        return costs

    def get_linear_system(
        self,
        trajs: torch.Tensor,
    ):
        A, b, K = None, None, None
        batch_size = trajs.shape[0]

        err_obst, H_obst = self.obst_factor.get_error(
            q_trajs=trajs,
            env=self.env,
            robot=self.robot,
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


class CostGP(Cost):
    def __init__(
        self, robot: Robot, n_support_points: int, start_state: torch.Tensor, dt: float, sigma_params: dict, **kwargs
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.start_state = start_state
        self.dt = dt

        self.sigma_start = sigma_params["sigma_start"]
        self.sigma_gp = sigma_params["sigma_gp"]

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.start_prior = UnaryFactor(
            self.dim,
            self.sigma_start,
            self.start_state,
            self.tensor_args,
        )

        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs: torch.Tensor):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)
        # Start cost
        err_p = self.start_prior.get_error(trajs[:, [0]], calc_jacobian=False)
        w_mat = self.start_prior.K
        start_costs = err_p @ w_mat.unsqueeze(0) @ err_p.transpose(1, 2)
        start_costs = start_costs.squeeze()

        # GP Trajectory cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1)
        gp_costs = gp_costs.squeeze()

        costs = start_costs + gp_costs

        return costs

    def get_linear_system(self, trajs: torch.Tensor):
        batch_size = trajs.shape[0]
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

        # Start prior factor
        err_p, H_p = self.start_prior.get_error(trajs[:, [0]])
        A[:, : self.dim, : self.dim] = H_p
        b[:, : self.dim] = err_p
        K[:, : self.dim, : self.dim] = self.start_prior.K

        # GP factors
        err_gp, H1_gp, H2_gp = self.gp_prior.get_error(trajs)

        A[:, self.dim :, : -self.dim] = torch.block_diag(*H1_gp)
        A[:, self.dim :, self.dim :] += torch.block_diag(*H2_gp)
        b[:, self.dim :] = einops.rearrange(err_gp, "b h d 1 -> b (h d) 1")
        K[:, self.dim :, self.dim :] += torch.block_diag(*self.gp_prior.Q_inv)

        # old code not vectorized
        # https://github.com/anindex/stoch_gpmp/blob/main/stoch_gpmp/costs/cost_functions.py#L161

        return A, b, K


class CostGPTrajectory(Cost):
    def __init__(self, robot: Robot, n_support_points: int, dt: float, sigma_gp: float = None, **kwargs):
        super().__init__(robot, n_support_points, **kwargs)
        self.dt = dt

        self.sigma_gp = sigma_gp

        self.set_cost_factors()

    def set_cost_factors(self):
        # ========= Cost factors ===============
        self.gp_prior = GPFactor(
            self.n_dof,
            self.sigma_gp,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

    def eval(self, trajs: torch.Tensor):
        # trajs = trajs.reshape(-1, self.n_support_points, self.dim)

        # GP cost
        err_gp = self.gp_prior.get_error(trajs, calc_jacobian=False)
        w_mat = self.gp_prior.Q_inv[0]  # repeated Q_inv
        w_mat = w_mat.reshape(1, 1, self.dim, self.dim)
        gp_costs = err_gp.transpose(2, 3) @ w_mat @ err_gp
        gp_costs = gp_costs.sum(1).squeeze()
        costs = gp_costs
        return costs

    def get_linear_system(self, trajs: torch.Tensor):
        pass


class CostGoalPrior(Cost):
    def __init__(
        self,
        robot: Robot,
        n_support_points: int,
        multi_goal_states: torch.Tensor = None,  # num_goal x n_dim (pos + vel)
        num_particles_per_goal: int = None,
        num_samples: int = None,
        sigma_goal_prior: float = None,
        **kwargs,
    ):
        super().__init__(robot, n_support_points, **kwargs)
        self.multi_goal_states = multi_goal_states
        self.num_goals = multi_goal_states.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = num_particles_per_goal * self.num_goals
        self.num_samples = num_samples
        self.sigma_goal_prior = sigma_goal_prior

        self.set_cost_factors()

    def set_cost_factors(self):
        self.multi_goal_prior = []
        # TODO: remove this for loop
        for i in range(self.num_goals):
            self.multi_goal_prior.append(
                UnaryFactor(
                    self.dim,
                    self.sigma_goal_prior,
                    self.multi_goal_states[i],
                    self.tensor_args,
                )
            )

    def eval(self, trajs: torch.Tensor):
        costs = 0
        if self.multi_goal_states is not None:
            x = trajs.reshape(
                self.num_goals,
                self.num_particles_per_goal * self.num_samples,
                self.n_support_points,
                self.dim,
            )
            costs = torch.zeros(
                self.num_goals,
                self.num_particles_per_goal * self.num_samples,
                **self.tensor_args,
            )
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g = self.multi_goal_prior[i].get_error(
                    x[i, :, [-1]], calc_jacobian=False
                )
                w_mat = self.multi_goal_prior[i].K
                goal_costs = err_g @ w_mat.unsqueeze(0) @ err_g.transpose(1, 2)
                goal_costs = goal_costs.squeeze()
                costs[i] += goal_costs
            costs = costs.flatten()
        return costs

    def get_linear_system(self, trajs: torch.Tensor):
        A, b, K = None, None, None
        if self.multi_goal_states is not None:
            npg = self.num_particles_per_goal
            batch_size = npg * self.num_goals
            x = trajs.reshape(
                self.num_goals,
                self.num_particles_per_goal,
                self.n_support_points,
                self.dim,
            )
            A = torch.zeros(
                batch_size,
                self.dim,
                self.dim * self.n_support_points,
                **self.tensor_args,
            )
            b = torch.zeros(batch_size, self.dim, 1, **self.tensor_args)
            K = torch.zeros(batch_size, self.dim, self.dim, **self.tensor_args)
            # TODO: remove this for loop
            for i in range(self.num_goals):
                err_g, H_g = self.multi_goal_prior[i].get_error(x[i, :, [-1]])
                A[i * npg : (i + 1) * npg, :, -self.dim :] = H_g
                b[i * npg : (i + 1) * npg] = err_g
                K[i * npg : (i + 1) * npg] = self.multi_goal_prior[i].K

        return A, b, K
