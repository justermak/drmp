from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from drmp.datasets.dataset import (
    TrajectoryDatasetBase,
    TrajectoryDatasetBSpline,
    TrajectoryDatasetDense,
)
from drmp.planning.guide import Guide, GuideSlow
from drmp.models.diffusion import DiffusionModelBase
from drmp.planning.costs import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
    CostJointVelocity,
)
from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.hybrid_planner import HybridPlanner
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.trajectory_utils import (
    create_straight_line_trajectory,
    get_trajectories_from_bsplines,
)


class ModelWrapperBase(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def sample(
        self,
        dataset: TrajectoryDatasetDense,
        data_normalized: Dict[str, Any],
        n_samples: int,
        debug: bool = False,
    ):
        pass


class DiffusionModelWrapper(ModelWrapperBase):
    def __init__(
        self,
        model: DiffusionModelBase,
        guide: Guide,
        t_start_guide: float,
        n_guide_steps: int,
        ddim: bool,
        use_extra_objects: bool,
    ):
        super().__init__(use_extra_objects)
        self.model = model
        self.guide = guide
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def sample(
        self,
        dataset: TrajectoryDatasetBase,
        data_normalized: Dict[str, Any],
        n_samples: int,
        debug: bool = False,
    ):
        context = self.model.build_context(data_normalized)
        hard_conditions = self.model.build_hard_conditions(data_normalized)

        trajectories_iters_normalized = self.model.run_inference(
            n_samples=n_samples,
            hard_conditions=hard_conditions,
            context=context,
            guide=self.guide,
            n_guide_steps=self.n_guide_steps,
            t_start_guide=self.t_start_guide,
            ddim=self.ddim,
        )

        trajectories_iters = dataset.normalizer.unnormalize(
            trajectories_iters_normalized
        )

        if isinstance(dataset, TrajectoryDatasetBSpline):
            trajectories_iters = get_trajectories_from_bsplines(
                control_points=trajectories_iters,
                n_support_points=dataset.n_support_points,
                degree=dataset.spline_degree,
            )

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
        dataset: TrajectoryDatasetDense,
        data_normalized: Dict[str, Any],
        n_samples: int,
        debug: bool = False,
    ):
        hard_conditions = {
            0: torch.cat(
                [
                    data_normalized["start_pos_normalized"],
                    torch.zeros_like(data_normalized["start_pos_normalized"]),
                ],
                dim=-1,
            ),
            dataset.n_support_points - 1: torch.cat(
                [
                    data_normalized["goal_pos_normalized"],
                    torch.zeros_like(data_normalized["goal_pos_normalized"]),
                ],
                dim=-1,
            ),
        }

        trajectories_iters_normalized = self.model.run_inference(
            context=None,
            hard_conditions=hard_conditions,
            n_samples=n_samples,
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
        dataset: TrajectoryDatasetBSpline,
        data_normalized: Dict[str, Any],
        n_samples: int,
        debug: bool = False,
    ):
        start = data_normalized["start_pos_normalized"]
        goal = data_normalized["goal_pos_normalized"]

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
            hard_conditions=hard_conditions,
            n_samples=n_samples,
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
        method: str,
        sample_steps: int,
        opt_steps: int,
    ):
        super().__init__(planner.use_extra_objects)
        self.planner = planner
        self.method = method
        self.sample_steps = sample_steps
        self.opt_steps = opt_steps

    def sample(
        self,
        dataset: TrajectoryDatasetDense,
        data_normalized: Dict[str, Any],
        n_samples: int,
        debug: bool = False,
    ):
        start_pos_normalized = data_normalized["start_pos_normalized"]
        goal_pos_normalized = data_normalized["goal_pos_normalized"]

        start_pos = dataset.normalizer.unnormalize(start_pos_normalized)
        goal_pos = dataset.normalizer.unnormalize(goal_pos_normalized)

        qs = torch.cat((start_pos.unsqueeze(0), goal_pos.unsqueeze(0)), dim=0)
        collision_mask = dataset.env.get_collision_mask(
            robot=self.planner.robot, qs=qs, on_extra=self.use_extra_objects
        )
        if collision_mask.any():
            return None, None

        self.planner.reset(start_pos, goal_pos)
        if self.method == "rrt-connect":
            trajectories_list = self.planner.optimize(
                sample_steps=self.sample_steps,
                debug=debug,
            )
            if all(t is None for t in trajectories_list):
                return None, None
            max_len = max([t.shape[0] for t in trajectories_list if t is not None])
            trajectories = torch.stack(
                [
                    torch.cat((t, goal_pos.repeat((max_len - t.shape[0], 1))), dim=0)
                    if t is not None
                    else torch.cat(
                        (start_pos, goal_pos.repeat((max_len - 1, 1))), dim=0
                    )
                    for t in trajectories_list
                ]
            )
            trajectories = torch.cat(
                (trajectories, torch.zeros_like(trajectories)), dim=-1
            )
        elif self.method == "gpmp2-uninformative":
            initial_trajectories = create_straight_line_trajectory(
                start_pos=start_pos,
                goal_pos=goal_pos,
                n_support_points=dataset.n_support_points,
                dt=dataset.robot.dt,
                tensor_args=dataset.tensor_args,
            ).repeat((n_samples, 1, 1))
            self.planner.reset_trajectories(initial_trajectories)
            trajectories = self.planner.optimize(
                opt_steps=self.opt_steps,
                debug=debug,
            )
        else:
            trajectories = self.planner.optimize(
                sample_steps=self.sample_steps,
                opt_steps=self.opt_steps,
                debug=debug,
            )

        trajectories_iters = None
        trajectories_final = trajectories

        return trajectories_iters, trajectories_final

    def cleanup(self):
        if hasattr(self.planner, "shutdown"):
            self.planner.shutdown()
        elif hasattr(self.planner, "sample_based_planner"):
            self.planner.sample_based_planner.shutdown()


class ModelConfigBase(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def prepare(
        self,
        dataset: TrajectoryDatasetDense,
        n_samples: int,
        tensor_args: Dict[str, Any],
    ) -> ModelWrapperBase:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class DiffusionConfig(ModelConfigBase):
    def __init__(
        self,
        model: DiffusionModelBase,
        use_extra_objects: bool,
        sigma_collision: float,
        sigma_gp: float,
        sigma_velocity: float,
        max_grad_norm: float,
        n_interpolate: int,
        t_start_guide: float,
        n_guide_steps: int,
        ddim: bool,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.model = model
        self.sigma_collision = sigma_collision
        self.sigma_gp = sigma_gp
        self.sigma_velocity = sigma_velocity
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.t_start_guide = t_start_guide
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def prepare(
        self,
        dataset: TrajectoryDatasetDense,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> DiffusionModelWrapper:
        collision_cost = None
        if self.sigma_collision is not None:
            collision_cost = CostCollision(
                robot=dataset.robot,
                env=dataset.env,
                n_support_points=dataset.n_support_points,
                sigma_collision=self.sigma_collision,
                use_extra_objects=self.use_extra_objects,
                tensor_args=tensor_args,
            )
        
        gp_cost = None
        if self.sigma_gp is not None:
            gp_cost = CostGPTrajectory(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                sigma_gp=self.sigma_gp,
                tensor_args=tensor_args,
            )
        
        velocity_cost = None
        if self.sigma_velocity is not None:
            velocity_cost = CostJointVelocity(
                robot=dataset.robot,
                n_support_points=dataset.n_support_points,
                sigma_velocity=self.sigma_velocity,
                tensor_args=tensor_args,
            )
            
        costs = [collision_cost, gp_cost, velocity_cost]

        cost = CostComposite(
            robot=dataset.robot,
            n_support_points=dataset.n_support_points,
            costs=costs,
            tensor_args=tensor_args,
        )

        guide = GuideSlow(
            dataset=dataset,
            cost=cost,
            max_grad_norm=self.max_grad_norm,
            n_interpolate=self.n_interpolate,
        )

        return DiffusionModelWrapper(
            model=self.model,
            guide=guide,
            t_start_guide=self.t_start_guide,
            n_guide_steps=self.n_guide_steps,
            ddim=self.ddim,
            use_extra_objects=self.use_extra_objects,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "diffusion",
            "use_extra_objects": self.use_extra_objects,
            "sigma_collision": self.sigma_collision,
            "sigma_gp": self.sigma_gp,
            "sigma_velocity": self.sigma_velocity,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "t_start_guide": self.t_start_guide,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
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
        dataset: TrajectoryDatasetDense,
        tensor_args: Dict[str, Any],
        n_samples: int,
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
                cost=cost,
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
            "algorithm": "mpd",
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
        dataset: TrajectoryDatasetDense,
        tensor_args: Dict[str, Any],
        n_samples: int,
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
            "algorithm": "mpd-splines",
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


class RRTConnectConfig(ModelConfigBase):
    def __init__(
        self,
        sample_steps: int,
        use_extra_objects: bool,
        rrt_connect_max_step_size: float,
        rrt_connect_max_radius: float,
        rrt_connect_n_samples: int,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.sample_steps = sample_steps
        self.rrt_connect_max_step_size = rrt_connect_max_step_size
        self.rrt_connect_max_radius = rrt_connect_max_radius
        self.rrt_connect_n_samples = rrt_connect_n_samples

    def prepare(
        self,
        dataset: TrajectoryDatasetDense,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> ClassicalPlannerWrapper:
        planner = RRTConnect(
            env=dataset.env,
            robot=dataset.generating_robot,
            max_step_size=self.rrt_connect_max_step_size,
            max_radius=self.rrt_connect_max_radius,
            n_samples=self.rrt_connect_n_samples,
            n_trajectories=n_samples,
            use_extra_objects=self.use_extra_objects,
            tensor_args=tensor_args,
        )

        wrapper = ClassicalPlannerWrapper(
            planner=planner,
            method="rrt-connect",
            sample_steps=self.sample_steps,
            opt_steps=0,
        )
        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "rrt_connect",
            "sample_steps": self.sample_steps,
            "use_extra_objects": self.use_extra_objects,
            "rrt_connect_max_step_size": self.rrt_connect_max_step_size,
            "rrt_connect_max_radius": self.rrt_connect_max_radius,
            "rrt_connect_n_samples": self.rrt_connect_n_samples,
        }


class GPMP2UninformativeConfig(ModelConfigBase):
    def __init__(
        self,
        opt_steps: int,
        use_extra_objects: bool,
        n_dim: int,
        gpmp2_n_interpolate: int,
        gpmp2_num_samples: int,
        gpmp2_sigma_start: float,
        gpmp2_sigma_goal_prior: float,
        gpmp2_sigma_gp: float,
        gpmp2_sigma_collision: float,
        gpmp2_step_size: float,
        gpmp2_delta: float,
        gpmp2_method: str,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.opt_steps = opt_steps
        self.n_dim = n_dim
        self.gpmp2_n_interpolate = gpmp2_n_interpolate
        self.gpmp2_num_samples = gpmp2_num_samples
        self.gpmp2_sigma_start = gpmp2_sigma_start
        self.gpmp2_sigma_goal_prior = gpmp2_sigma_goal_prior
        self.gpmp2_sigma_gp = gpmp2_sigma_gp
        self.gpmp2_sigma_collision = gpmp2_sigma_collision
        self.gpmp2_step_size = gpmp2_step_size
        self.gpmp2_delta = gpmp2_delta
        self.gpmp2_method = gpmp2_method

    def prepare(
        self,
        dataset: TrajectoryDatasetDense,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> ClassicalPlannerWrapper:
        planner = GPMP2(
            robot=dataset.generating_robot,
            n_dim=self.n_dim,
            n_trajectories=n_samples,
            env=dataset.env,
            tensor_args=tensor_args,
            n_support_points=dataset.n_support_points,
            dt=dataset.generating_robot.dt,
            n_interpolate=self.gpmp2_n_interpolate,
            num_samples=self.gpmp2_num_samples,
            sigma_start=self.gpmp2_sigma_start,
            sigma_gp=self.gpmp2_sigma_gp,
            sigma_goal_prior=self.gpmp2_sigma_goal_prior,
            sigma_collision=self.gpmp2_sigma_collision,
            step_size=self.gpmp2_step_size,
            delta=self.gpmp2_delta,
            method=self.gpmp2_method,
            use_extra_objects=self.use_extra_objects,
        )

        wrapper = ClassicalPlannerWrapper(
            planner=planner,
            method="gpmp2-uninformative",
            sample_steps=0,
            opt_steps=self.opt_steps,
        )
        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "gpmp2_uninformative",
            "opt_steps": self.opt_steps,
            "use_extra_objects": self.use_extra_objects,
            "n_dim": self.n_dim,
            "gpmp2_n_interpolate": self.gpmp2_n_interpolate,
            "gpmp2_num_samples": self.gpmp2_num_samples,
            "gpmp2_sigma_start": self.gpmp2_sigma_start,
            "gpmp2_sigma_goal_prior": self.gpmp2_sigma_goal_prior,
            "gpmp2_sigma_gp": self.gpmp2_sigma_gp,
            "gpmp2_sigma_collision": self.gpmp2_sigma_collision,
            "gpmp2_step_size": self.gpmp2_step_size,
            "gpmp2_delta": self.gpmp2_delta,
            "gpmp2_method": self.gpmp2_method,
        }


class GPMP2RRTPriorConfig(ModelConfigBase):
    def __init__(
        self,
        sample_steps: int,
        opt_steps: int,
        use_extra_objects: bool,
        n_dim: int,
        rrt_connect_max_step_size: float,
        rrt_connect_max_radius: float,
        rrt_connect_n_samples: int,
        gpmp2_n_interpolate: int,
        gpmp2_num_samples: int,
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
        self.gpmp2_num_samples = gpmp2_num_samples
        self.gpmp2_sigma_start = gpmp2_sigma_start
        self.gpmp2_sigma_goal_prior = gpmp2_sigma_goal_prior
        self.gpmp2_sigma_gp = gpmp2_sigma_gp
        self.gpmp2_sigma_collision = gpmp2_sigma_collision
        self.gpmp2_step_size = gpmp2_step_size
        self.gpmp2_delta = gpmp2_delta
        self.gpmp2_method = gpmp2_method

    def prepare(
        self,
        dataset: TrajectoryDatasetDense,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> ClassicalPlannerWrapper:
        sample_based_planner = RRTConnect(
            env=dataset.env,
            robot=dataset.generating_robot,
            max_step_size=self.rrt_connect_max_step_size,
            max_radius=self.rrt_connect_max_radius,
            n_samples=self.rrt_connect_n_samples,
            n_trajectories=n_samples,
            use_extra_objects=self.use_extra_objects,
            tensor_args=tensor_args,
        )

        optimization_based_planner = GPMP2(
            robot=dataset.generating_robot,
            n_dim=self.n_dim,
            n_trajectories=n_samples,
            env=dataset.env,
            tensor_args=tensor_args,
            n_support_points=dataset.n_support_points,
            dt=dataset.generating_robot.dt,
            n_interpolate=self.gpmp2_n_interpolate,
            num_samples=self.gpmp2_num_samples,
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
            tensor_args=tensor_args,
        )

        wrapper = ClassicalPlannerWrapper(
            planner=planner,
            method="gpmp2-rrt-prior",
            sample_steps=self.sample_steps,
            opt_steps=self.opt_steps,
        )
        return wrapper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "gpmp2_rrt_prior",
            "sample_steps": self.sample_steps,
            "opt_steps": self.opt_steps,
            "use_extra_objects": self.use_extra_objects,
            "n_dim": self.n_dim,
            "rrt_connect_max_step_size": self.rrt_connect_max_step_size,
            "rrt_connect_max_radius": self.rrt_connect_max_radius,
            "rrt_connect_n_samples": self.rrt_connect_n_samples,
            "gpmp2_n_interpolate": self.gpmp2_n_interpolate,
            "gpmp2_num_samples": self.gpmp2_num_samples,
            "gpmp2_sigma_start": self.gpmp2_sigma_start,
            "gpmp2_sigma_goal_prior": self.gpmp2_sigma_goal_prior,
            "gpmp2_sigma_gp": self.gpmp2_sigma_gp,
            "gpmp2_sigma_collision": self.gpmp2_sigma_collision,
            "gpmp2_step_size": self.gpmp2_step_size,
            "gpmp2_delta": self.gpmp2_delta,
            "gpmp2_method": self.gpmp2_method,
        }
