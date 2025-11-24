"""Training logs and logging functions."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from mpd.models import build_context


def log(
    train_step,
    model,
    datasubset,
    train_losses=None,
    val_losses=None,
    prefix="",
    debug=False,
    tensorboard_writer=None,
):
    """
    Generate and log trajectory generation logs to TensorBoard.

    This function:
    1. Logs training and validation losses
    2. Samples trajectories from the model
    3. Computes collision and success metrics
    4. Renders trajectory visualizations
    5. Logs scalars and figures to TensorBoard

    Args:
        train_step: Current training step
        model: The diffusion model
        datasubset: Training or validation subset
        train_losses: Dictionary of training losses {loss_name: loss_value}
        val_losses: Dictionary of validation losses {loss_name: loss_value}
        prefix: Prefix for logged metrics (e.g., 'TRAINING ' or 'VALIDATION ')
        debug: If True, displays plots interactively
        tensorboard_writer: TensorBoard SummaryWriter instance
    """
    model.eval()
    with torch.no_grad():
        # ------------------------------------------------------------------------------------
        # Log losses to TensorBoard
        if tensorboard_writer is not None:
            # Log training losses
            if train_losses is not None:
                for loss_name, loss_value in train_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{prefix}{loss_name}", loss_value, train_step
                    )

            # Log validation losses
            if val_losses is not None:
                for loss_name, loss_value in val_losses.items():
                    tensorboard_writer.add_scalar(
                        f"{loss_name}",  # Already has 'VALIDATION ' prefix
                        loss_value,
                        train_step,
                    )

        dataset = datasubset.dataset

        # ------------------------------------------------------------------------------------
        # Get a random task from the dataset
        trajectory_id = np.random.choice(datasubset.indices)
        task_id = dataset.map_trajectory_id_to_task_id[trajectory_id]

        data_normalized = dataset[trajectory_id]
        context = build_context(model, dataset, data_normalized)

        # ------------------------------------------------------------------------------------
        # Sample trajectories with the diffusion model
        n_samples = 25
        horizon = dataset.n_support_points
        hard_conds = data_normalized["hard_conds"]

        trajs_normalized = model.run_inference(
            context,
            hard_conds,
            n_samples=n_samples,
            horizon=horizon,
            deterministic_steps=0,
        )

        # unnormalize trajectory samples from the diffusion model
        trajs = dataset.unnormalize(trajs_normalized, dataset.field_key_traj)

        # ------------------------------------------------------------------------------------
        # STATISTICS - Log metrics to TensorBoard
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(
                f"{prefix}percentage_free_trajs",
                dataset.task.compute_fraction_free_trajs(trajs),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}percentage_collision_intensity",
                dataset.task.compute_collision_intensity_trajs(trajs),
                train_step,
            )
            tensorboard_writer.add_scalar(
                f"{prefix}success",
                dataset.task.compute_success_free_trajs(trajs),
                train_step,
            )

        # ------------------------------------------------------------------------------------
        # Render visualizations

        # Dataset trajectory
        fig_joint_trajs_dataset, _, fig_robot_trajs_dataset, _ = dataset.render(
            task_id=task_id, render_joint_trajectories=True
        )

        # Diffusion trajectory
        pos_trajs = dataset.robot.get_position(trajs)
        start_state_pos = pos_trajs[0][0]
        goal_state_pos = pos_trajs[0][-1]

        fig_joint_trajs_diffusion, _ = (
            dataset.planner_visualizer.plot_joint_space_state_trajectories(
                trajs=pos_trajs,
                pos_start_state=start_state_pos,
                pos_goal_state=goal_state_pos,
                vel_start_state=torch.zeros_like(start_state_pos),
                vel_goal_state=torch.zeros_like(goal_state_pos),
                linestyle="dashed",
            )
        )

        fig_robot_trajs_diffusion = None
        # fig_robot_trajs_diffusion, _ = dataset.planner_visualizer.render_robot_trajectories(
        #     trajs=pos_trajs, start_state=start_state_pos, goal_state=goal_state_pos,
        #     linestyle='dashed'
        # )

        # Log figures to TensorBoard
        if tensorboard_writer is not None:
            if fig_joint_trajs_dataset is not None:
                tensorboard_writer.add_figure(
                    f"{prefix}joint_trajectories_DATASET",
                    fig_joint_trajs_dataset,
                    train_step,
                )
            if fig_robot_trajs_dataset is not None:
                tensorboard_writer.add_figure(
                    f"{prefix}robot_trajectories_DATASET",
                    fig_robot_trajs_dataset,
                    train_step,
                )

            if fig_joint_trajs_diffusion is not None:
                tensorboard_writer.add_figure(
                    f"{prefix}joint_trajectories_DIFFUSION",
                    fig_joint_trajs_diffusion,
                    train_step,
                )
            if fig_robot_trajs_diffusion is not None:
                tensorboard_writer.add_figure(
                    f"{prefix}robot_trajectories_DIFFUSION",
                    fig_robot_trajs_diffusion,
                    train_step,
                )

        # Show plots interactively if in debug mode
        if debug:
            plt.show()

        # Clean up figures
        if fig_joint_trajs_dataset is not None:
            plt.close(fig_joint_trajs_dataset)
        if fig_robot_trajs_dataset is not None:
            plt.close(fig_robot_trajs_dataset)
        if fig_joint_trajs_diffusion is not None:
            plt.close(fig_joint_trajs_diffusion)
        if fig_robot_trajs_diffusion is not None:
            plt.close(fig_robot_trajs_diffusion)

    model.train()
