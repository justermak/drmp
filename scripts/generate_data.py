import os

import configargparse
import torch

from drmp.config import DEFAULT_DATA_GENERATION_ARGS
from drmp.dataset.dataset import TrajectoryDataset
from drmp.utils import fix_random_seed


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print("-------- GENERATING DATA --------")
    print(f"env: {args.env_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"N trajectories per task: {args.n_trajectories_per_task}")

    dataset = TrajectoryDataset(
        env_name=args.env_name,
        robot_name=args.robot_name,
        robot_margin=args.robot_margin,
        generating_robot_margin=args.generating_robot_margin,
        n_support_points=args.n_support_points,
        duration=args.duration,
        spline_degree=args.spline_degree,
        tensor_args=tensor_args,
    )

    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)

    dataset.generate_data(
        dataset_dir=dataset_dir,
        n_tasks=args.n_tasks,
        n_trajectories_per_task=args.n_trajectories_per_task,
        threshold_start_goal_pos=args.threshold_start_goal_pos,
        n_sampling_steps=args.n_sampling_steps,
        n_optimization_steps=args.n_optimization_steps,
        use_gpmp2=args.use_gpmp2,
        n_control_points=args.n_control_points,
        val_portion=args.val_portion,
        rrt_connect_max_step_size=args.rrt_connect_max_step_size,
        rrt_connect_max_radius=args.rrt_connect_max_radius,
        rrt_connect_n_samples=args.rrt_connect_n_samples,
        gpmp2_n_interpolate=args.gpmp2_n_interpolate,
        gpmp2_sigma_start=args.gpmp2_sigma_start,
        gpmp2_sigma_goal_prior=args.gpmp2_sigma_goal_prior,
        gpmp2_sigma_gp=args.gpmp2_sigma_gp,
        gpmp2_sigma_collision=args.gpmp2_sigma_collision,
        gpmp2_step_size=args.gpmp2_step_size,
        gpmp2_delta=args.gpmp2_delta,
        gpmp2_method=args.gpmp2_method,
        grad_lambda_obstacles=args.grad_lambda_obstacles,
        grad_lambda_position=args.grad_lambda_position,
        grad_lambda_velocity=args.grad_lambda_velocity,
        grad_lambda_acceleration=args.grad_lambda_acceleration,
        grad_max_grad_norm=args.grad_max_grad_norm,
        grad_n_interpolate=args.grad_n_interpolate,
        n_processes=args.n_processes,
        seed=args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {}

    for key, value in DEFAULT_DATA_GENERATION_ARGS.items():
        arg_name = f"--{key}"
        arg_type = type(value) if value is not None else str

        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no_{key}", dest=key, action="store_false")
        else:
            kwargs = {"type": arg_type, "default": value}
            if key in special_args:
                kwargs.update(special_args[key])
            parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()
    run(args)
