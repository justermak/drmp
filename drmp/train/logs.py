import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from drmp.datasets.dataset import TrajectoryDataset
from drmp.models.models import PlanningModel
from drmp.utils.visualizer import Visualizer


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
        hard_conds = model.build_hard_conds(data_normalized)

        trajs_normalized = model.run_inference(
            context,
            hard_conds,
            n_samples=25,
        )[-1]

        trajs = dataset.trajs_normalizer.unnormalize(trajs_normalized)

        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                f"{prefix}percentage_free_trajs",
                dataset.task.compute_fraction_free_trajs(trajs),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}percentage_collision_intensity",
                dataset.task.compute_collision_intensity(trajs),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}success",
                dataset.task.compute_success_free_trajs(trajs),
                train_step,
            )

        planning_visualizer = Visualizer(dataset.task)

        fig, ax = planning_visualizer.render_robot_trajectories(
            trajs=trajs,
        )

        if fig is not None and tensorboard_writer is not None:
            tensorboard_writer.add_figure(
                f"{prefix}joint_trajectories_DIFFUSION",
                fig,
                train_step,
            )

        if debug:
            plt.show()

        if fig is not None:
            plt.close(fig)

    model.train()
