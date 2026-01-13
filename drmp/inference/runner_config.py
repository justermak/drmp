from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference.guides import GuideTrajectories
from drmp.models.diffusion import PlanningModel
from drmp.planning.costs.cost_functions import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
)
from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.planning.planners.gpmp2 import GPMP2
from drmp.planning.planners.hybrid_planner import HybridPlanner
from drmp.planning.planners.parallel_sample_based_planner import (
    ParallelSampleBasedPlanner,
)
from drmp.planning.planners.rrt_connect import RRTConnect
from drmp.utils.trajectory_utils import create_straight_line_trajectory


class BaseRunnerModelWrapper(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects

    @abstractmethod
    def sample(
        self,
        dataset: TrajectoryDataset,
        data_normalized: Dict[str, Any],
        n_samples: int,
    ):
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class DiffusionModelWrapper(BaseRunnerModelWrapper):
    def __init__(
        self,
        model: PlanningModel,
        guide: GuideTrajectories,
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
        data_normalized: Dict[str, Any],
        n_samples: int,
    ):
        context = self.model.build_context(data_normalized)
        hard_conds = self.model.build_hard_conditions(data_normalized)

        trajectories_iters_normalized = self.model.run_inference(
            context=context,
            hard_conds=hard_conds,
            n_samples=n_samples,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            guide=self.guide,
            n_guide_steps=self.n_guide_steps,
            ddim=self.ddim,
        )

        trajectories_iters = dataset.normalizer.unnormalize(
            trajectories_iters_normalized
        )
        trajectories_final = trajectories_iters[-1]

        return trajectories_iters, trajectories_final

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
        }


class LegacyDiffusionModelWrapper(BaseRunnerModelWrapper):
    def __init__(
        self,
        model: Any,
        guide: Optional[GuideTrajectories],
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
        data_normalized: Dict[str, Any],
        n_samples: int,
    ):
        hard_conds = {
            0: data_normalized["start_states_normalized"],
            dataset.n_support_points - 1: data_normalized["goal_states_normalized"],
        }

        trajectories_iters_normalized = self.model.run_inference(
            context=None,
            hard_conds=hard_conds,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
        }


class ClassicalPlannerWrapper(BaseRunnerModelWrapper):
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
        dataset: TrajectoryDataset,
        data_normalized: Dict[str, Any],
        n_samples: int,
    ):
        start_states_normalized = data_normalized["start_states_normalized"]
        goal_states_normalized = data_normalized["goal_states_normalized"]

        start_states = dataset.normalizer.unnormalize(start_states_normalized)
        goal_states = dataset.normalizer.unnormalize(goal_states_normalized)

        start_pos = dataset.robot.get_position(start_states)
        goal_pos = dataset.robot.get_position(goal_states)

        qs = torch.cat((start_states.unsqueeze(0), goal_states.unsqueeze(0)), dim=0)
        collision_mask = dataset.env.get_collision_mask(
            robot=self.planner.robot, qs=qs, on_extra=self.use_extra_objects
        )
        if collision_mask.any():
            return None, None

        self.planner.reset(start_pos, goal_pos)
        if self.method == "rrt-connect":
            trajectories_list = self.planner.optimize(
                sample_steps=self.sample_steps,
                debug=False,
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
                debug=False,
            )
        else:
            trajectories = self.planner.optimize(
                sample_steps=self.sample_steps,
                opt_steps=self.opt_steps,
                debug=False,
            )

        trajectories_iters = None
        trajectories_final = trajectories

        return trajectories_iters, trajectories_final

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "sample_steps": self.sample_steps,
            "opt_steps": self.opt_steps,
            "smoothen": self.smoothen,
        }

    def cleanup(self):
        if hasattr(self.planner, "shutdown"):
            self.planner.shutdown()
        elif hasattr(self.planner, "sample_based_planner"):
            self.planner.sample_based_planner.shutdown()


class BaseRunnerConfig(ABC):
    def __init__(self, use_extra_objects: bool):
        self.use_extra_objects = use_extra_objects
    
    @abstractmethod
    def prepare(
        self,
        dataset: TrajectoryDataset,
        n_samples: int,
        tensor_args: Dict[str, Any],
    ) -> BaseRunnerModelWrapper:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


class DiffusionRunnerConfig(BaseRunnerConfig):
    def __init__(
        self,
        model: PlanningModel,
        use_extra_objects: bool,
        sigma_collision: float,
        sigma_gp: float,
        do_clip_grad: bool,
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
        self.do_clip_grad = do_clip_grad
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> DiffusionModelWrapper:
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

        guide = GuideTrajectories(
            dataset=dataset,
            cost=cost,
            do_clip_grad=self.do_clip_grad,
            max_grad_norm=self.max_grad_norm,
            n_interpolate=self.n_interpolate,
        )

        return DiffusionModelWrapper(
            model=self.model,
            guide=guide,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
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
            "do_clip_grad": self.do_clip_grad,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
        }


class LegacyDiffusionRunnerConfig(BaseRunnerConfig):
    def __init__(
        self,
        model: Any,
        use_extra_objects: bool,
        sigma_collision: float,
        sigma_gp: float,
        do_clip_grad: bool,
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
        self.do_clip_grad = do_clip_grad
        self.max_grad_norm = max_grad_norm
        self.n_interpolate = n_interpolate
        self.start_guide_steps_fraction = start_guide_steps_fraction
        self.n_guide_steps = n_guide_steps
        self.ddim = ddim

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> LegacyDiffusionModelWrapper:
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

            guide = GuideTrajectories(
                dataset=dataset,
                cost=cost,
                do_clip_grad=self.do_clip_grad,
                max_grad_norm=self.max_grad_norm,
                n_interpolate=self.n_interpolate,
            )

        return LegacyDiffusionModelWrapper(
            model=self.model,
            guide=guide,
            start_guide_steps_fraction=self.start_guide_steps_fraction,
            n_guide_steps=self.n_guide_steps,
            ddim=self.ddim,
            use_extra_objects=self.use_extra_objects,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": "legacy_diffusion",
            "use_extra_objects": self.use_extra_objects,
            "sigma_collision": self.sigma_collision,
            "sigma_gp": self.sigma_gp,
            "do_clip_grad": self.do_clip_grad,
            "max_grad_norm": self.max_grad_norm,
            "n_interpolate": self.n_interpolate,
            "start_guide_steps_fraction": self.start_guide_steps_fraction,
            "n_guide_steps": self.n_guide_steps,
            "ddim": self.ddim,
        }


class RRTConnectRunnerConfig(BaseRunnerConfig):
    def __init__(
        self,
        sample_steps: int,
        use_parallel: bool,
        max_processes: int,
        use_extra_objects: bool,
        rrt_connect_step_size: float,
        rrt_connect_n_radius: float,
        rrt_connect_n_samples: int,
        seed: int,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.sample_steps = sample_steps
        self.use_parallel = use_parallel
        self.max_processes = max_processes
        self.rrt_connect_step_size = rrt_connect_step_size
        self.rrt_connect_n_radius = rrt_connect_n_radius
        self.rrt_connect_n_samples = rrt_connect_n_samples
        self.seed = seed

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> ClassicalPlannerWrapper:
        rrt_planner = RRTConnect(
            env=dataset.env,
            robot=dataset.robot,
            tensor_args=tensor_args,
            step_size=self.rrt_connect_step_size,
            n_radius=self.rrt_connect_n_radius,
            n_samples=self.rrt_connect_n_samples,
            use_extra_objects=self.use_extra_objects,
        )
        planner = ParallelSampleBasedPlanner(
            planner=rrt_planner,
            n_trajectories=n_samples,
            use_parallel=self.use_parallel,
            max_processes=self.max_processes,
            seed=self.seed,
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
            "use_parallel": self.use_parallel,
            "max_processes": self.max_processes,
            "use_extra_objects": self.use_extra_objects,
            "rrt_connect_step_size": self.rrt_connect_step_size,
            "rrt_connect_n_radius": self.rrt_connect_n_radius,
            "rrt_connect_n_samples": self.rrt_connect_n_samples,
            "seed": self.seed,
        }


class GPMP2UninformativeRunnerConfig(BaseRunnerConfig):
    def __init__(
        self,
        opt_steps: int,
        use_extra_objects: bool,
        n_dof: int,
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
        self.n_dof = n_dof
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
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> ClassicalPlannerWrapper:
        planner = GPMP2(
            robot=dataset.generating_robot,
            n_dof=self.n_dof,
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
            "n_dof": self.n_dof,
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


class GPMP2RRTPriorRunnerConfig(BaseRunnerConfig):
    def __init__(
        self,
        sample_steps: int,
        opt_steps: int,
        use_parallel: bool,
        max_processes: int,
        use_extra_objects: bool,
        n_dof: int,
        rrt_connect_step_size: float,
        rrt_connect_n_radius: float,
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
        seed: int,
    ):
        super().__init__(use_extra_objects=use_extra_objects)
        self.sample_steps = sample_steps
        self.opt_steps = opt_steps
        self.use_parallel = use_parallel
        self.max_processes = max_processes
        self.n_dof = n_dof
        self.rrt_connect_step_size = rrt_connect_step_size
        self.rrt_connect_n_radius = rrt_connect_n_radius
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
        self.seed = seed

    def prepare(
        self,
        dataset: TrajectoryDataset,
        tensor_args: Dict[str, Any],
        n_samples: int,
    ) -> ClassicalPlannerWrapper:
        rrt_planner = RRTConnect(
            env=dataset.env,
            robot=dataset.generating_robot,
            tensor_args=tensor_args,
            step_size=self.rrt_connect_step_size,
            n_radius=self.rrt_connect_n_radius,
            n_samples=self.rrt_connect_n_samples,
            use_extra_objects=self.use_extra_objects,
        )
        sample_based_planner = ParallelSampleBasedPlanner(
            planner=rrt_planner,
            n_trajectories=n_samples,
            use_parallel=self.use_parallel,
            max_processes=self.max_processes,
            seed=self.seed,
        )

        gpmp2_planner = GPMP2(
            robot=dataset.generating_robot,
            n_dof=self.n_dof,
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
            opt_based_planner=gpmp2_planner,
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
            "use_parallel": self.use_parallel,
            "max_processes": self.max_processes,
            "use_extra_objects": self.use_extra_objects,
            "n_dof": self.n_dof,
            "rrt_connect_step_size": self.rrt_connect_step_size,
            "rrt_connect_n_radius": self.rrt_connect_n_radius,
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
            "seed": self.seed,
        }
