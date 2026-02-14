from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from drmp.dataset.dataset import TrajectoryDataset
from drmp.model.generative_models import GenerativeModel
from drmp.planning.costs import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
    CostJointAcceleration,
    CostJointPosition,
    CostJointVelocity,
    CostObstacles,
)
from drmp.planning.guide import Guide
from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.hybrid_planner import HybridPlanner
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils import get_trajectories_from_bsplines


class ModelWrapperBase(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        pass


class GenerativeModelWrapper(ModelWrapperBase):
    def __init__(
        self,
        use_extra_objects: bool,
        model: GenerativeModel,
        model_name: str,
        guide: Guide,
        t_start_guide: float,
        n_guide_steps: int,
        additional_args: Dict[str, Any],
    ):
        super().__init__(use_extra_objects)
        self.model = model
        self.model_name = model_name
        self.guide = guide
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.additional_args = additional_args
        
    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        context = self.model.build_context(data)
        start_pos = data["start_pos"]
        goal_pos = data["goal_pos"]
        
        trajectories_iters_normalized = self.model.run_inference(
            n_samples=n_trajectories_per_task,
            context=context,
            guide=self.guide,
            n_guide_steps=self.n_guide_steps,
            t_start_guide=self.t_start_guide,
            debug=debug,
            **self.additional_args,
        )

        trajectories_iters = dataset.normalizer.unnormalize(
            trajectories_iters_normalized
        )

        if dataset.n_control_points is not None:
            trajectories_iters[..., :2, :] = start_pos.unsqueeze(0)
            trajectories_iters[..., -2:, :] = goal_pos.unsqueeze(0)
            trajectories_iters = get_trajectories_from_bsplines(
                control_points=trajectories_iters,
                n_support_points=dataset.n_support_points,
                degree=dataset.robot.spline_degree,
            )
        else:
            trajectories_iters[..., 0, :] = start_pos.unsqueeze(0)
            trajectories_iters[..., -1, :] = goal_pos.unsqueeze(0)

        trajectories_final = trajectories_iters[-1]

        return trajectories_iters, trajectories_final


class MPDModelWrapper(ModelWrapperBase):
    def __init__(
        self,
        model: Any,
        guide: Guide,
        start_guide_steps_fraction: float,
        n_guide_steps: int,
        ddim: bool,
        use_extra_objects: bool,
    ):
        super().__init__(use_extra_objects)
        self.model = model
        self.guide = guide
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        hard_conditions = {
            0: torch.cat(
                [
                    data["start_pos_normalized"],
                    torch.zeros_like(data["start_pos_normalized"]),
                ],
                dim=-1,
            ),
            dataset.n_support_points - 1: torch.cat(
                [
                    data["goal_pos_normalized"],
                    torch.zeros_like(data["goal_pos_normalized"]),
                ],
                dim=-1,
            ),
        }

        trajectories_iters_normalized = self.model.run_inference(
            context=None,
            hard_conds=hard_conditions,
            n_samples=n_trajectories_per_task,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            guide=self.guide,
            n_guide_steps=self.n_guide_steps,
            horizon=dataset.n_support_points,
            return_chain=True,
            ddim=self.ddim,
        )

        trajectories_iters = dataset.normalizer.unnormalize(
            trajectories_iters_normalized
        )
        trajectories_final = trajectories_iters[-1]

        return trajectories_iters, trajectories_final


class MPDSplinesModelWrapper(ModelWrapperBase):
    def __init__(
        self,
        model: Any,
        start_guide_steps_fraction: float,
        guide: Guide,
        n_guide_steps: int,
        ddim: bool,
        use_extra_objects: bool,
        guide_lr: float,
        scale_grad_prior: float,
        ddim_sampling_timesteps: int,
    ):
        super().__init__(use_extra_objects)
        self.model = model
        self.guide = guide
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim
        self.guide_lr = guide_lr
        self.scale_grad_prior = scale_grad_prior
        self.ddim_sampling_timesteps = ddim_sampling_timesteps

    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        start = data["start_pos_normalized"]
        goal = data["goal_pos_normalized"]

        horizon = dataset.n_control_points

        context = {
            "qs_normalized": torch.cat((start, goal), dim=-1),
        }

        hard_conditions = {
            0: start,
            1: start,
            horizon - 2: goal,
            horizon - 1: goal,
        }

        t_start_guide = int(
            self.start_guide_steps_fraction * self.ddim_sampling_timesteps
        )

        control_points_iters = self.model.run_inference(
            guide=self.guide,
            context_d=context,
            hard_conds=hard_conditions,
            n_samples=n_trajectories_per_task,
            horizon=horizon,
            return_chain=True,
            return_chain_x_recon=False,
            results_ns=None,
            ddim_scale_grad_prior=self.scale_grad_prior,
            ddim_sampling_timesteps=self.ddim_sampling_timesteps,
            guide_lr=self.guide_lr,
            n_guide_steps=self.n_guide_steps,
            t_start_guide=t_start_guide,
            debug=False,
        )

        trajectories_iters = get_trajectories_from_bsplines(
            control_points=control_points_iters,
            n_support_points=dataset.n_support_points,
            degree=dataset.spline_degree,
        )

        return trajectories_iters, trajectories_iters[-1]


class ClassicalPlannerWrapper(ModelWrapperBase):
    def __init__(
        self,
        planner: ClassicalPlanner,
        sample_steps: int,
        opt_steps: int,
    ):
        super().__init__(planner.use_extra_objects)
        self.planner = planner
        self.sample_steps = sample_steps
        self.opt_steps = opt_steps

    def sample(
        self,
        dataset: TrajectoryDataset,
        data: Dict[str, Any],
        n_trajectories_per_task: int,
        debug: bool = False,
    ):
        start_pos = data["start_pos"]
        goal_pos = data["goal_pos"]

        qs = torch.cat((start_pos.unsqueeze(0), goal_pos.unsqueeze(0)), dim=0)
        collision_mask = dataset.robot.get_collision_mask(
            env=dataset.env, qs=qs, on_extra=self.use_extra_objects
        )
        if collision_mask.any():
            return None, None

        self.planner.reset(start_pos, goal_pos)
        
        trajectories = self.planner.optimize(
            sample_steps=self.sample_steps,
            opt_steps=self.opt_steps,
            debug=debug,
        )

        trajectories_iters = None
        trajectories_final = trajectories

        return trajectories_iters, trajectories_final


class ModelConfigBase(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def prepare(
        self,
        dataset: TrajectoryDataset,
        n_trajectories_per_task: int,
        tensor_args: Dict[str, Any],
    ) -> ModelWrapperBase:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class GenerativeModelConfig(ModelConfigBase):
    def __init__(
        self,
        model: GenerativeModel,
        model_name: str,
        t_start_guide: float,
        n_guide_steps: int,
        use_extra_objects: bool,
        lambda_obstacles: float,
        lambda_position: float,
        lambda_velocity: float,
        lambda_acceleration: float,
        max_grad_norm: float,
        n_interpolate: int,
        additional_args: Dict[str, Any],
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.model = model
        self.model_name = model_name
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.lambda_obstacles = lambda_obstacles
        self.lambda_position = lambda_position
        self.lambda_velocity = lambda_velocity
        self.lambda_acceleration = lambda_acceleration
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.additional_args = additional_args

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> GenerativeModelWrapper:
        collision_cost = None
        if self.lambda_obstacles is not None:
            collision_cost = CostObstacles(
                robot=dataset.robot,
                env=dataset.env,
                n_support_points=dataset.n_support_points,
                lambda_obstacles=self.lambda_obstacles,
                use_extra_objects=self.use_extra_objects,
                tensor_args=tensor_args,
            )

        position_cost = None
        if self.lambda_position is not None:
            position_cost = CostJointPosition(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                lambda_position=self.lambda_position,
                tensor_args=tensor_args,
            )

        velocity_cost = None
        if self.lambda_velocity is not None:
            velocity_cost = CostJointVelocity(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                lambda_velocity=self.lambda_velocity,
                tensor_args=tensor_args,
            )

        acceleration_cost = None
        if self.lambda_acceleration is not None:
            acceleration_cost = CostJointAcceleration(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                lambda_acceleration=self.lambda_acceleration,
                tensor_args=tensor_args,
            )

        costs = [
            cost
            for cost in [
                collision_cost,
                position_cost,
                velocity_cost,
                acceleration_cost,
            ]
            if cost is not None
        ]

        guide = (
            Guide(
                dataset=dataset,
                costs=costs,
                max_grad_norm=self.max_grad_norm,
                n_interpolate=self.n_interpolate,
            )
        )

        return GenerativeModelWrapper(
            use_extra_objects=self.use_extra_objects,
            model=self.model,
            model_name=self.model_name,
            guide=guide,
            t_start_guide=self.t_start_guide,
            n_guide_steps=self.n_guide_steps,
            additional_args=self.additional_args,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_extra_objects": self.use_extra_objects,
            "lambda_obstacles": self.lambda_obstacles,
            "lambda_position": self.lambda_position,
            "lambda_velocity": self.lambda_velocity,
            "lambda_acceleration": self.lambda_acceleration,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "t_start_guide": self.t_start_guide,
            "n_guide_steps": self.n_guide_steps,
            "additional_args": self.additional_args,
        }


class MPDConfig(ModelConfigBase):
    def __init__(
        self,
        model: Any,
        use_extra_objects: bool,
        sigma_collision: float,
        sigma_gp: float,
        max_grad_norm: float,
        n_interpolate: int,
        start_guide_steps_fraction: float,
        n_guide_steps: int,
        ddim: bool,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.model = model
        self.sigma_collision = sigma_collision
        self.sigma_gp = sigma_gp
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> MPDModelWrapper:
        guide = None
        if self.n_guide_steps > 0:
            collision_costs = [
                CostCollision(
                    robot=dataset.robot,
                    env=dataset.env,
                    n_support_points=dataset.n_support_points,
                    sigma_collision=self.sigma_collision,
                    use_extra_objects=self.use_extra_objects,
                    tensor_args=tensor_args,
                )
            ]

            sharpness_costs = [
                CostGPTrajectory(
                    robot=dataset.robot,
                    n_support_points=dataset.n_support_points,
                    sigma_gp=self.sigma_gp,
                    tensor_args=tensor_args,
                )
            ]

            costs = collision_costs + sharpness_costs

            cost = CostComposite(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                costs=costs,
                tensor_args=tensor_args,
            )

            guide = Guide(
                dataset=dataset,
                costs=[cost],
                max_grad_norm=self.max_grad_norm,
                n_interpolate=self.n_interpolate,
            )

        return MPDModelWrapper(
            model=self.model,
            guide=guide,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            n_guide_steps=self.n_guide_steps,
            ddim=self.ddim,
            use_extra_objects=self.use_extra_objects,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_extra_objects": self.use_extra_objects,
            "sigma_collision": self.sigma_collision,
            "sigma_gp": self.sigma_gp,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
        }


class MPDSplinesConfig(ModelConfigBase):
    def __init__(
        self,
        model: Any,
        start_guide_steps_fraction: float,
        n_guide_steps: int,
        ddim: bool,
        guide_lr: float,
        scale_grad_prior: float,
        sigma_collision: float,
        sigma_gp: float,
        max_grad_norm: float,
        n_interpolate: int,
        ddim_sampling_timesteps: int,
        use_extra_objects: bool,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.model = model
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim
        self.guide_lr = guide_lr
        self.scale_grad_prior = scale_grad_prior
        self.sigma_collision = sigma_collision
        self.sigma_gp = sigma_gp
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.ddim_sampling_timesteps = ddim_sampling_timesteps

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> MPDSplinesModelWrapper:
        guide = None
        if self.n_guide_steps > 0:
            collision_costs = [
                CostCollision(
                    robot=dataset.robot,
                    env=dataset.env,
                    n_support_points=dataset.n_support_points,
                    sigma_collision=self.sigma_collision,
                    use_extra_objects=self.use_extra_objects,
                    tensor_args=tensor_args,
                )
            ]

            sharpness_costs = [
                CostGPTrajectory(
                    robot=dataset.robot,
                    n_support_points=dataset.n_support_points,
                    sigma_gp=self.sigma_gp,
                    tensor_args=tensor_args,
                )
            ]

            costs = collision_costs + sharpness_costs

            cost = CostComposite(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                costs=costs,
                tensor_args=tensor_args,
            )

            guide = Guide(
                dataset=dataset,
                cost=cost,
                max_grad_norm=self.max_grad_norm,
                n_interpolate=self.n_interpolate,
            )

        return MPDSplinesModelWrapper(
            model=self.model,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            guide=guide,
            n_guide_steps=self.n_guide_steps,
            ddim=self.ddim,
            use_extra_objects=self.use_extra_objects,
            guide_lr=self.guide_lr,
            scale_grad_prior=self.scale_grad_prior,
            ddim_sampling_timesteps=self.ddim_sampling_timesteps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
            "guide_lr": self.guide_lr,
            "scale_grad_prior": self.scale_grad_prior,
            "sigma_collision": self.sigma_collision,
            "sigma_gp": self.sigma_gp,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "ddim_sampling_timesteps": self.ddim_sampling_timesteps,
        }


class ClassicalConfig(ModelConfigBase):
    def __init__(
        self,
        use_extra_objects: bool,
        sample_steps: int,
        opt_steps: int,
        n_dim: int,
        rrt_connect_max_step_size: float,
        rrt_connect_max_radius: float,
        rrt_connect_n_samples: int,
        gpmp2_n_interpolate: int,
        gpmp2_sigma_start: float,
        gpmp2_sigma_goal_prior: float,
        gpmp2_sigma_gp: float,
        gpmp2_sigma_collision: float,
        gpmp2_step_size: float,
        gpmp2_delta: float,
        gpmp2_method: str,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.sample_steps = sample_steps
        self.opt_steps = opt_steps
        self.n_dim = n_dim
        self.rrt_connect_max_step_size = rrt_connect_max_step_size
        self.rrt_connect_max_radius = rrt_connect_max_radius
        self.rrt_connect_n_samples = rrt_connect_n_samples
        self.gpmp2_n_interpolate = gpmp2_n_interpolate
        self.gpmp2_sigma_start = gpmp2_sigma_start
        self.gpmp2_sigma_goal_prior = gpmp2_sigma_goal_prior
        self.gpmp2_sigma_gp = gpmp2_sigma_gp
        self.gpmp2_sigma_collision = gpmp2_sigma_collision
        self.gpmp2_step_size = gpmp2_step_size
        self.gpmp2_delta = gpmp2_delta
        self.gpmp2_method = gpmp2_method
        
    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_trajectories_per_task: int,
    ) -> ClassicalPlannerWrapper:
        sample_based_planner = None
        if self.sample_steps is not None:
            sample_based_planner = RRTConnect(
                env=dataset.env,
                robot=dataset.generating_robot,
                tensor_args=tensor_args,
                n_trajectories=n_trajectories_per_task,
                max_step_size=self.rrt_connect_max_step_size,
                max_radius=self.rrt_connect_max_radius,
                n_samples=self.rrt_connect_n_samples,
                use_extra_objects=self.use_extra_objects,
            )

        optimization_based_planner = None
        if self.opt_steps is not None:
            optimization_based_planner = GPMP2(
                robot=dataset.generating_robot,
                n_dim=dataset.generating_robot.n_dim,
                n_trajectories=n_trajectories_per_task,
                env=dataset.env,
                tensor_args=tensor_args,
                n_support_points=dataset.n_support_points,
                dt=dataset.generating_robot.dt,
                n_interpolate=self.gpmp2_n_interpolate,
                sigma_start=self.gpmp2_sigma_start,
                sigma_gp=self.gpmp2_sigma_gp,
                sigma_goal_prior=self.gpmp2_sigma_goal_prior,
                sigma_collision=self.gpmp2_sigma_collision,
                step_size=self.gpmp2_step_size,
                delta=self.gpmp2_delta,
                method=self.gpmp2_method,
                use_extra_objects=self.use_extra_objects,
            )

        planner = HybridPlanner(
            sample_based_planner=sample_based_planner,
            optimization_based_planner=optimization_based_planner,
            n_support_points=dataset.n_support_points,
            dt=dataset.generating_robot.dt,
            tensor_args=tensor_args,
        )

        wrapper = ClassicalPlannerWrapper(
            planner=planner,
            sample_steps=self.sample_steps,
            opt_steps=self.opt_steps,
        )
        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_steps": self.sample_steps,
            "opt_steps": self.opt_steps,
            "use_extra_objects": self.use_extra_objects,
            "n_dim": self.n_dim,
            "rrt_connect_max_step_size": self.rrt_connect_max_step_size,
            "rrt_connect_max_radius": self.rrt_connect_max_radius,
            "rrt_connect_n_samples": self.rrt_connect_n_samples,
            "gpmp2_n_interpolate": self.gpmp2_n_interpolate,
            "gpmp2_sigma_start": self.gpmp2_sigma_start,
            "gpmp2_sigma_goal_prior": self.gpmp2_sigma_goal_prior,
            "gpmp2_sigma_gp": self.gpmp2_sigma_gp,
            "gpmp2_sigma_collision": self.gpmp2_sigma_collision,
            "gpmp2_step_size": self.gpmp2_step_size,
            "gpmp2_delta": self.gpmp2_delta,
            "gpmp2_method": self.gpmp2_method,
        }
