import torch
from typing import Any, Dict
from drmp.config import DEFAULT_TENSOR_ARGS


from drmp.utils.torch_timer import TimerCUDA
from drmp.planning.costs.cost_functions import (
    Cost,
    CostCollision,
    CostComposite,
    CostGoalPrior,
    CostGP,
)
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


def build_gpmp2_cost_composite(
    robot:Robot,
    n_support_points: int,
    dt: float,
    start_state: torch.Tensor,
    multi_goal_states: torch.Tensor,
    num_particles_per_goal: int,
    sigma_start:float,
    sigma_gp:float,
    sigma_coll: float,
    sigma_goal_prior: float,
    num_samples: int,
    env: EnvBase,
    on_extra: bool = False,
    tensor_args: Dict[str, Any]=DEFAULT_TENSOR_ARGS,
) -> Cost:
    """
    Construct cost composite function for GPMP and StochGPMP
    """
    cost_func_list = []

    # Start state + GP cost
    cost_sigmas = dict(
        sigma_start=sigma_start,
        sigma_gp=sigma_gp,
    )
    start_state_zero_vel = torch.cat(
        (start_state, torch.zeros(start_state.nelement(), **tensor_args))
    )
    cost_gp_prior = CostGP(
        robot,
        n_support_points,
        start_state_zero_vel,
        dt,
        cost_sigmas,
        tensor_args=tensor_args,
    )
    cost_func_list.append(cost_gp_prior)

    # Goal state cost
    if multi_goal_states is not None:
        multi_goal_states_zero_vel = torch.cat(
            (multi_goal_states, torch.zeros_like(multi_goal_states)), dim=-1
        ).unsqueeze(0)  # add batch dim for interface
        cost_goal_prior = CostGoalPrior(
            robot,
            n_support_points,
            multi_goal_states=multi_goal_states_zero_vel,
            num_particles_per_goal=num_particles_per_goal,
            num_samples=num_samples,
            sigma_goal_prior=sigma_goal_prior,
            tensor_args=tensor_args,
        )
        cost_func_list.append(cost_goal_prior)

    # Collision costs
    cost_collision = CostCollision(
        robot=robot,
        env=env,
        n_support_points=n_support_points,
        sigma_coll=sigma_coll,
        on_extra=on_extra,
        tensor_args=tensor_args,
    )
    cost_func_list.append(cost_collision)

    cost_composite = CostComposite(
        robot, n_support_points, cost_func_list, tensor_args=tensor_args
    )
    return cost_composite


class GPMP2():
    def __init__(
        self,
        robot: Robot,
        env: EnvBase,
        n_dof: int,
        n_support_points: int,
        num_particles_per_goal: int,
        dt: float,
        start_pos: torch.Tensor,
        multi_goal_pos: torch.Tensor,
        num_samples:int,
        sigma_start:float,
        sigma_gp:float,
        sigma_goal_prior:float,
        sigma_coll: float,
        step_size: float,
        delta: float,
        method: str,
        on_extra: bool = False,
        tensor_args: Dict[str, Any]=DEFAULT_TENSOR_ARGS,
    ):
        self.tensor_args = tensor_args
        self.n_dof = n_dof
        self.dim = 2 * self.n_dof
        self.n_support_points = n_support_points
        self.N = self.dim * self.n_support_points
        self.num_goals = multi_goal_pos.shape[0]
        self.num_particles_per_goal = num_particles_per_goal
        self.num_particles = self.num_goals * self.num_particles_per_goal
        self.dt = dt
        self.step_size = step_size
        self.delta = delta
        self.method = method
        
        self.start_state = torch.cat(
            [start_pos, torch.zeros_like(start_pos)], dim=-1
        )
        self.multi_goal_states = torch.cat(
            [multi_goal_pos, torch.zeros_like(multi_goal_pos)],
            dim=-1,
        )

        self.cost = build_gpmp2_cost_composite(
                robot=robot,
                n_support_points=n_support_points,
                dt=dt,
                start_state=start_pos,
                multi_goal_states=multi_goal_pos,
                num_particles_per_goal=num_particles_per_goal,
                env=env,
                on_extra=on_extra,
                sigma_start=sigma_start,
                sigma_gp=sigma_gp,
                sigma_coll=sigma_coll,
                sigma_goal_prior=sigma_goal_prior,
                num_samples=num_samples,
                tensor_args=tensor_args,
            )
        
        self._particle_means: torch.Tensor = None
        
        
    def get_trajs(self):
        trajs = self._particle_means.clone()
        return trajs

    def get_costs(self, state_trajectories, **observation):
        costs = self.cost(state_trajectories, **observation)
        return costs


    def reset(self, initial_particle_means: torch.Tensor):
        self._particle_means = initial_particle_means

    def optimize(self, opt_iters: int = 1, print_freq: int = 100, debug: bool=False):
        self.opt_iters = opt_iters
        with TimerCUDA() as t_opt:
            for opt_step in range(opt_iters):
                b, K = self._step()
                if debug and opt_step % print_freq == 0:
                    costs_temp = self.get_costs(b, K)
                    self.print_info(opt_step + 1, t_opt.elapsed, costs_temp)
                        
            if debug:
                self.print_info(opt_iters, t_opt.elapsed, self.get_costs(b, K))
        trajs = self.get_trajs()
        return trajs
    
    def _step(self):
        A, b, K = self.cost.get_linear_system(
            self._particle_means,
        )

        J_t_J, g = self._get_grad_terms(
            A,
            b,
            K,
            delta=self.delta,
        )

        d_theta = self.get_torch_solve(
            J_t_J,
            g,
            method=self.method,
        )

        d_theta = d_theta.view(
            self.num_particles,
            self.n_support_points,
            self.dim,
        )

        self._particle_means = self._particle_means + self.step_size * d_theta
        self._particle_means.detach_()

        return b, K

    def _get_grad_terms(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        K: torch.Tensor,
        delta: float = 0.0,
    ):
        # Levenberg - Marquardt approximation
        # Original implementation with dense matrices
        I = torch.eye(
            self.N, self.N, device=self.tensor_args["device"], dtype=A.dtype
        )
        A_t_K = A.transpose(-2, -1) @ K
        A_t_A = A_t_K @ A
        
        # J_t_J = A_t_A + delta * I * torch.diagonal(A_t_A, dim1=-2, dim2=-1).unsqueeze(-1)
        # Since hessian will be averaged over particles, add diagonal matrix of the mean.
        diag_A_t_A = A_t_A.mean(0) * I
        J_t_J = A_t_A + delta * diag_A_t_A
        g = A_t_K @ b

        return J_t_J, g

    def get_torch_solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        method: str,
    ):
        if method == "inverse":
            res = torch.linalg.solve(A, b)
        elif method == "cholesky":
            l, _ = torch.linalg.cholesky_ex(A)
            res = torch.cholesky_solve(b, l)

        elif method == "lstq":
            # empirically slower
            res = torch.linalg.lstsq(A, b)[0]
        else:
            raise NotImplementedError

        return res

    def get_costs(self, errors: torch.Tensor, w_mat: torch.Tensor):
        costs = errors.transpose(1, 2) @ w_mat.unsqueeze(0) @ errors
        return costs.reshape(
            self.num_particles,
        )

    def print_info(self, iteration: int, t: float, costs: torch.Tensor):
        pad = len(str(self.opt_iters))
        mean_cost = costs.mean().item()
        min_cost = costs.min().item()
        max_cost = costs.max().item()
        print(
            f"Iteration: {iteration:>{pad}}/{self.opt_iters:>{pad}} "
            f"| Time: {t:.3f}s "
            f"| Cost (mean/min/max): {mean_cost:.3e}/{min_cost:.3e}/{max_cost:.3e}"
        )

