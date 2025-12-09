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
    compute_sharpness,
    compute_success,
    compute_waypoints_variance,
)


def log(
    step: int,
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
                        f"{prefix}{loss_name}", loss_value, step
                    )

            if val_losses is not None:
                for loss_name, loss_value in val_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}",
                        loss_value,
                        step,
                    )
        dataset: TrajectoryDataset = subset.dataset
        trajectory_id = np.random.choice(subset.indices)
        data_normalized = dataset[trajectory_id]

        context = model.build_context(data_normalized)
        hard_conds = model.build_hard_conditions(data_normalized)

        trajectories_normalized = model.run_inference(
            context,
            hard_conds,
            n_samples=25,
        )[-1]

        trajectories = dataset.normalizer.unnormalize(trajectories_normalized)
        trajectories_collision, trajectories_free, trajectories_collision_mask = dataset.env.get_trajectories_collision_and_free(trajectories=trajectories, robot=dataset.robot)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                f"{prefix}free_fraction",
                compute_free_fraction(trajectories_free, trajectories_collision),
                step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}collision_intensity",
                compute_collision_intensity(trajectories_collision_mask),
                step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}success",
                compute_success(trajectories_free),
                step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}avg_path_length",
                compute_path_length(trajectories_free, dataset.robot).mean(),
                step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}avg_smoothness",
                compute_sharpness(trajectories_free, dataset.robot).mean(),
                step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}waypoints_variance",
                compute_waypoints_variance(trajectories_free, dataset.robot),
                step,
            )

        planning_visualizer = Visualizer(dataset.env, dataset.robot)

        fig, ax = planning_visualizer.render_scene(
            trajectories=trajectories,
            start_state=trajectories[0, 0],
            goal_state=trajectories[0, -1],
            save_path=None,
        )

        if fig is not None and tensorboard_writer is not None:
            tensorboard_writer.add_figure(
                f"{prefix}trajectories_figure",
                fig,
                step,
            )

        if debug:
            plt.show()

        if fig is not None:
            plt.close(fig)

    model.train()
