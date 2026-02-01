import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from drmp.datasets.dataset import TrajectoryDatasetBase, TrajectoryDatasetBSpline
from drmp.models.diffusion import DiffusionModelBase, DiffusionSplinesShortcut
from drmp.planning.guide import Guide
from drmp.planning.metrics import (
    compute_collision_intensity,
    compute_free_fraction,
    compute_path_length,
    compute_sharpness,
    compute_success,
    compute_waypoints_variance,
)
from drmp.utils.trajectory import get_trajectories_from_bsplines
from drmp.utils.visualizer import Visualizer


def _log_trajectories_metrics(
    model: DiffusionModelBase,
    context: torch.Tensor,
    hard_conditions: dict,
    dataset: TrajectoryDatasetBase,
    planning_visualizer: Visualizer,
    tensorboard_writer: SummaryWriter,
    prefix: str,
    suffix: str,
    step: int,
    ddim: bool,
    shortcut_steps: int,
    guide=None,
    use_extra_objects: bool = None,
    t_start_guide: float = None,
    n_guide_steps: int = None,
) -> None:
    if isinstance(model, DiffusionSplinesShortcut):
        trajectories_normalized = model.run_inference(
            n_samples=20,
            hard_conditions=hard_conditions,
            context=context,
            guide=guide,
            n_guide_steps=n_guide_steps,
            t_start_guide=t_start_guide,
            shortcut_steps=shortcut_steps,
        )[-1]
    else:
        trajectories_normalized = model.run_inference(
            n_samples=20,
            hard_conditions=hard_conditions,
            context=context,
            guide=guide,
            n_guide_steps=n_guide_steps,
            t_start_guide=t_start_guide,
            ddim=ddim,
        )[-1]

    trajectories = dataset.normalizer.unnormalize(trajectories_normalized)

    if isinstance(dataset, TrajectoryDatasetBSpline):
        trajectories = get_trajectories_from_bsplines(
            control_points=trajectories,
            n_support_points=dataset.n_support_points,
            degree=dataset.robot.spline_degree,
        )

    trajectories_collision, trajectories_free, trajectories_collision_mask = (
        dataset.env.get_trajectories_collision_and_free(
            trajectories=trajectories, robot=dataset.robot, on_extra=use_extra_objects
        )
    )

    tensorboard_writer.add_scalar(
        f"{prefix}free_fraction{suffix}",
        compute_free_fraction(trajectories_free, trajectories_collision),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}collision_intensity{suffix}",
        compute_collision_intensity(trajectories_collision_mask),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}success{suffix}",
        compute_success(trajectories_free),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}avg_path_length{suffix}",
        compute_path_length(trajectories_free, dataset.robot).mean(),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}avg_smoothness{suffix}",
        compute_sharpness(trajectories_free, dataset.robot).mean(),
        step,
    )
    tensorboard_writer.add_scalar(
        f"{prefix}waypoints_variance{suffix}",
        compute_waypoints_variance(trajectories_free, dataset.robot),
        step,
    )

    fig, ax = planning_visualizer.render_scene(
        trajectories=trajectories,
        start_pos=trajectories[0, 0],
        goal_pos=trajectories[0, -1],
        save_path=None,
    )

    tensorboard_writer.add_figure(
        f"{prefix}trajectories_figure{suffix}",
        fig,
        step,
    )

    plt.close(fig)


def log(
    step: int,
    model: DiffusionModelBase,
    subset: Subset,
    prefix: str,
    tensorboard_writer: SummaryWriter,
    guide: Guide,
    guide_extra: Guide,
    ddim: bool,
    shortcut_steps: int,
    t_start_guide: float,
    n_guide_steps: int,
    train_losses: dict = None,
    val_losses: dict = None,
    debug: bool = False,
):
    model.eval()
    with torch.no_grad():
        if tensorboard_writer is not None:
            if train_losses is not None:
                for loss_name, loss_value in train_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}", loss_value, step
                    )

            if val_losses is not None:
                for loss_name, loss_value in val_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}",
                        loss_value,
                        step,
                    )

        dataset: TrajectoryDatasetBase = subset.dataset
        trajectory_id = np.random.choice(subset.indices)
        data_normalized = dataset[trajectory_id]

        context = model.build_context(data_normalized)
        hard_conditions = model.build_hard_conditions(data_normalized)

        planning_visualizer = Visualizer(
            env=dataset.env, robot=dataset.robot, use_extra_objects=False
        )
        planning_visualizer_extra = Visualizer(
            env=dataset.env, robot=dataset.robot, use_extra_objects=False
        )

        if tensorboard_writer is not None:
            _log_trajectories_metrics(
                model=model,
                context=context,
                hard_conditions=hard_conditions,
                dataset=dataset,
                planning_visualizer=planning_visualizer,
                tensorboard_writer=tensorboard_writer,
                prefix=prefix,
                suffix="",
                step=step,
                ddim=ddim,
                shortcut_steps=shortcut_steps,
                guide=None,
            )

        if guide is not None and tensorboard_writer is not None:
            _log_trajectories_metrics(
                model=model,
                context=context,
                hard_conditions=hard_conditions,
                dataset=dataset,
                planning_visualizer=planning_visualizer,
                tensorboard_writer=tensorboard_writer,
                prefix=prefix,
                suffix="_guide",
                step=step,
                ddim=ddim,
                shortcut_steps=shortcut_steps,
                guide=guide,
                t_start_guide=t_start_guide,
                n_guide_steps=n_guide_steps,
            )

        if guide_extra is not None and tensorboard_writer is not None:
            _log_trajectories_metrics(
                model=model,
                context=context,
                hard_conditions=hard_conditions,
                dataset=dataset,
                planning_visualizer=planning_visualizer_extra,
                tensorboard_writer=tensorboard_writer,
                prefix=prefix,
                suffix="_guide_extra",
                step=step,
                ddim=ddim,
                shortcut_steps=shortcut_steps,
                guide=guide_extra,
                use_extra_objects=True,
                t_start_guide=t_start_guide,
                n_guide_steps=n_guide_steps,
            )
