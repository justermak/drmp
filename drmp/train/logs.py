import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from drmp.datasets.dataset import TrajectoryDataset
from drmp.models.diffusion import PlanningModel
from drmp.utils.visualizer import Visualizer
from drmp.planning.metrics import (
    compute_collision_intensity,
    compute_free_fraction,
    compute_path_length,
    compute_smoothness,
    compute_success,
    compute_waypoints_variance,
)


def log(
    train_step: int,
    model: PlanningModel,
    subset: Subset,
    train_losses: dict = None,
    val_losses: dict = None,
    prefix: str = "",
    tensorboard_writer: SummaryWriter = None,
    debug: bool = False,
):
    model.eval()
    with torch.no_grad():
        if tensorboard_writer is not None:
            if train_losses is not None:
                for loss_name, loss_value in train_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}", loss_value, train_step
                    )

            if val_losses is not None:
                for loss_name, loss_value in val_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}",
                        loss_value,
                        train_step,
                    )
        dataset: TrajectoryDataset = subset.dataset
        trajectory_id = np.random.choice(subset.indices)
        data_normalized = dataset[trajectory_id]

        context = model.build_context(data_normalized)
        hard_conds = model.build_hard_conditions(data_normalized)

        trajs_normalized = model.run_inference(
            context,
            hard_conds,
            n_samples=25,
        )[-1]

        trajs = dataset.normalizer.unnormalize(trajs_normalized)
        trajs_collision, trajs_free, trajs_collision_mask = dataset.env.get_trajs_collision_and_free(dataset.robot, trajs)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                f"{prefix}free_fraction",
                compute_free_fraction(trajs_free, trajs_collision),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}collision_intensity",
                compute_collision_intensity(trajs_collision_mask),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}success",
                compute_success(trajs_free),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}avg_path_length",
                compute_path_length(trajs_free, dataset.robot).mean(),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}avg_smoothness",
                compute_smoothness(trajs_free, dataset.robot).mean(),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}waypoints_variance",
                compute_waypoints_variance(trajs_free, dataset.robot),
                train_step,
            )

        planning_visualizer = Visualizer(dataset.env, dataset.robot)

        fig, ax = planning_visualizer.render_scene(
            trajs=trajs,
            start_state=trajs[0, 0],
            goal_state=trajs[0, -1],
        )

        if fig is not None and tensorboard_writer is not None:
            tensorboard_writer.add_figure(
                f"{prefix}trajectories_figure",
                fig,
                train_step,
            )

        if debug:
            plt.show()

        if fig is not None:
            plt.close(fig)

    model.train()
