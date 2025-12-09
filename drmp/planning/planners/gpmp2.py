from typing import Any, Dict

import torch

from drmp.config import DEFAULT_TENSOR_ARGS
from drmp.planning.costs.cost_functions import (
    Cost,
    CostCollision,
    CostComposite,
    CostGoalPrior,
    CostGP,
)
from drmp.utils.torch_timer import TimerCUDA
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


def build_gpmp2_cost_composite(
    robot: Robot,
    env: EnvBase,
    n_support_points: int,
    start_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    n_trajectories: int,
    sigma_start: float,
    sigma_goal_prior: float,
    sigma_gp: float,
    sigma_collision: float,
    num_samples: int,
    tensor_args: Dict[str, Any],
    use_extra_obstacles: bool = False,
) -> Cost:
    costs = []

    start_state = torch.cat(
        (start_pos, torch.zeros(start_pos.nelement(), **tensor_args))
    )
    cost_gp_prior = CostGP(
        robot=robot,
        n_support_points=n_support_points,
        start_state=start_state,
        sigma_start=sigma_start,
        sigma_gp=sigma_gp,
        tensor_args=tensor_args,
    )
    costs.append(cost_gp_prior)

    goal_state = torch.cat((goal_pos, torch.zeros_like(goal_pos)), dim=-1)
    cost_goal_prior = CostGoalPrior(
        robot=robot,
        n_support_points=n_support_points,
        goal_state=goal_state,
        n_trajectories=n_trajectories,
        num_samples=num_samples,
        sigma_goal_prior=sigma_goal_prior,
        tensor_args=tensor_args,
    )
    costs.append(cost_goal_prior)

    cost_collision = CostCollision(
        robot=robot,
        env=env,
        n_support_points=n_support_points,
        sigma_collision=sigma_collision,
        use_extra_obstacles=use_extra_obstacles,
        tensor_args=tensor_args,
    )
    costs.append(cost_collision)

    cost_composite = CostComposite(
        robot=robot,
        n_support_points=n_support_points,
        costs=costs,
        tensor_args=tensor_args,
    )
    return cost_composite


class GPMP2:
    def __init__(
        self,
        robot: Robot,
        env: EnvBase,
        n_dof: int,
        n_support_points: int,
        n_trajectories: int,
        dt: float,
        n_interpolate: int,
        num_samples: int,
        sigma_start: float,
        sigma_gp: float,
        sigma_goal_prior: float,
        sigma_collision: float,
        step_size: float,
        delta: float,
        method: str,
        use_extra_obstacles: bool = False,
        tensor_args: Dict[str, Any] = DEFAULT_TENSOR_ARGS,
    ):
        self.tensor_args = tensor_args
        self.robot = robot
        self.env = env
        self.n_dof = n_dof
        self.dim = 2 * self.n_dof
        self.n_support_points = n_support_points
        self.N = self.dim * self.n_support_points
        self.n_trajectories = n_trajectories
        self.dt = dt
        self.delta = delta
        self.method = method
        self.use_extra_obstacles = use_extra_obstacles
        
        self.n_interpolate = n_interpolate
        self.num_samples = num_samples
        self.sigma_start = sigma_start
        self.sigma_gp = sigma_gp
        self.sigma_goal_prior = sigma_goal_prior
        self.sigma_collision = sigma_collision
        self.step_size = step_size

        self._particle_means: torch.Tensor = None

    def _build_start_goal_cost(self, start_pos: torch.Tensor, goal_pos: torch.Tensor):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.start_state = torch.cat([start_pos, torch.zeros_like(start_pos)], dim=-1)
        self.goal_states = torch.cat([goal_pos, torch.zeros_like(goal_pos)], dim=-1)
        self.cost = build_gpmp2_cost_composite(
            robot=self.robot,
            env=self.env,
            n_support_points=self.n_support_points,
            start_pos=start_pos,
            goal_pos=goal_pos,
            n_trajectories=self.n_trajectories,
            use_extra_obstacles=self.use_extra_obstacles,
            sigma_start=self.sigma_start,
            sigma_gp=self.sigma_gp,
            sigma_collision=self.sigma_collision,
            sigma_goal_prior=self.sigma_goal_prior,
            num_samples=self.num_samples,
            tensor_args=self.tensor_args,
        )

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        self.start_pos = start_pos.to(**self.tensor_args)
        self.goal_pos = goal_pos.to(**self.tensor_args)
        self._build_start_goal_cost(self.start_pos, self.goal_pos)
        self._particle_means = None

    def reset_trajectories(self, initial_particle_means: torch.Tensor):
        self._particle_means = initial_particle_means

    def get_trajectories(self):
        trajectories = self._particle_means.clone()
        return trajectories

    def optimize(self, opt_steps: int = 1, print_freq: int = 100, debug: bool = True):
        self.opt_steps = opt_steps
        with TimerCUDA() as t_opt:
            for opt_step in range(opt_steps):
                b, K = self._step()
                if debug and opt_step % print_freq == 0:
                    self.print_info(opt_step + 1, t_opt.elapsed, self.get_costs(b, K))

            if debug:
                self.print_info(opt_steps, t_opt.elapsed, self.get_costs(b, K))

        trajectories = self.get_trajectories()
        return trajectories

    def _step(self):
        A, b, K = self.cost.get_linear_system(
            trajectories=self._particle_means, n_interpolate=self.n_interpolate
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
            self.n_trajectories,
            self.n_support_points,
            self.dim,
        )

        self._particle_means = (
            self._particle_means + self.step_size * d_theta
        ).detach()

        return b, K

    def _get_grad_terms(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        K: torch.Tensor,
        delta: float = 0.0,
    ):
        I = torch.eye(self.N, self.N, device=self.tensor_args["device"], dtype=A.dtype)
        A_t_K = A.transpose(-2, -1) @ K
        A_t_A = A_t_K @ A

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
            self.n_trajectories,
        )

    def print_info(self, step: int, t: float, costs: torch.Tensor):
        pad = len(str(self.opt_steps))
        mean_cost = costs.mean().item()
        min_cost = costs.min().item()
        max_cost = costs.max().item()
        print(
            f"Step: {step:>{pad}}/{self.opt_steps:>{pad}} "
            f"| Time: {t:.3f}s "
            f"| Cost (mean/min/max): {mean_cost:.3e}/{min_cost:.3e}/{max_cost:.3e}"
        )
