import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_INFERENCE_CLASSICAL_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference.runner import run_inference, create_test_subset
from drmp.inference.runner_config import (
    RRTConnectRunnerConfig,
    GPMP2UninformativeRunnerConfig,
    GPMP2RRTPriorRunnerConfig,
)
from drmp.utils.torch_utils import fix_random_seed
from drmp.utils.yaml import load_config_from_yaml


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}
    
    print("-------- CLASSICAL PLANNING INFERENCE STARTED --------")
    print(f"method: {args.method}")
    print(f"dataset: {args.dataset_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_samples: {args.n_samples}")
    
    # Load dataset
    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_config_path = os.path.join(dataset_dir, "config.yaml")
    dataset_config = load_config_from_yaml(dataset_config_path)
    
    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "env_name": dataset_config["env_name"],
        "normalizer_name": dataset_config["normalizer_name"],
        "robot_margin": dataset_config["robot_margin"],
        "cutoff_margin": dataset_config["cutoff_margin"],
        "n_support_points": dataset_config["n_support_points"],
        "duration": dataset_config["duration"],
        "tensor_args": tensor_args,
    }
    dataset = TrajectoryDataset(**dataset_init_config)
    dataset.load_data()
    train_subset, _, val_subset, _ = dataset.load_train_val_split()
    
    splits = eval(args.splits)
    
    # Generate test subset if needed
    test_subset = None
    if "test" in splits:
        test_subset = create_test_subset(
            dataset=dataset,
            n_tasks=args.n_tasks,
            threshold_start_goal_pos=args.threshold_start_goal_pos,
            tensor_args=tensor_args,
        )
        if test_subset is None:
            return
    
    # Create experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.method}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    # Create the appropriate runner config based on method
    if args.method == "rrt-connect":
        runner_config = RRTConnectRunnerConfig(
            sample_steps=args.sample_steps,
            use_parallel=args.use_parallel,
            max_processes=args.max_processes,
            smoothen=args.smoothen,
            use_extra_objects=args.use_extra_objects,
            rrt_connect_step_size=args.rrt_connect_step_size,
            rrt_connect_n_radius=args.rrt_connect_n_radius,
            rrt_connect_n_samples=args.rrt_connect_n_samples,
            seed=args.seed,
        )
        
    elif args.method == "gpmp2-uninformative":
        runner_config = GPMP2UninformativeRunnerConfig(
            opt_steps=args.opt_steps,
            use_extra_objects=args.use_extra_objects,
            n_dof=args.n_dof,
            gpmp2_n_interpolate=args.gpmp2_n_interpolate,
            gpmp2_num_samples=args.gpmp2_num_samples,
            gpmp2_sigma_start=args.gpmp2_sigma_start,
            gpmp2_sigma_goal_prior=args.gpmp2_sigma_goal_prior,
            gpmp2_sigma_gp=args.gpmp2_sigma_gp,
            gpmp2_sigma_collision=args.gpmp2_sigma_collision,
            gpmp2_step_size=args.gpmp2_step_size,
            gpmp2_delta=args.gpmp2_delta,
            gpmp2_method=args.gpmp2_method,
        )
        
    elif args.method == "gpmp2-rrt-prior":
        runner_config = GPMP2RRTPriorRunnerConfig(
            sample_steps=args.sample_steps,
            opt_steps=args.opt_steps,
            use_parallel=args.use_parallel,
            max_processes=args.max_processes,
            use_extra_objects=args.use_extra_objects,
            n_dof=args.n_dof,
            rrt_connect_step_size=args.rrt_connect_step_size,
            rrt_connect_n_radius=args.rrt_connect_n_radius,
            rrt_connect_n_samples=args.rrt_connect_n_samples,
            gpmp2_n_interpolate=args.gpmp2_n_interpolate,
            gpmp2_num_samples=args.gpmp2_num_samples,
            gpmp2_sigma_start=args.gpmp2_sigma_start,
            gpmp2_sigma_goal_prior=args.gpmp2_sigma_goal_prior,
            gpmp2_sigma_gp=args.gpmp2_sigma_gp,
            gpmp2_sigma_collision=args.gpmp2_sigma_collision,
            gpmp2_step_size=args.gpmp2_step_size,
            gpmp2_delta=args.gpmp2_delta,
            gpmp2_method=args.gpmp2_method,
            seed=args.seed,
        )
        
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    run_inference(
        runner_config=runner_config,
        dataset=dataset,
        train_subset=train_subset if "train" in splits else None,
        val_subset=val_subset if "val" in splits else None,
        test_subset=test_subset,
        generations_dir=args.generations_dir,
        experiment_name=experiment_name,
        n_tasks=args.n_tasks,
        n_samples=args.n_samples,
        debug=args.debug,
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    
    for key, value in DEFAULT_INFERENCE_CLASSICAL_ARGS.items():
        arg_name = f"--{key}"
        arg_type = type(value if value is not None else str)
        
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no_{key}", dest=key, action="store_false")
        else:
            kwargs = {"type": arg_type, "default": value}
            parser.add_argument(arg_name, **kwargs)
    
    args = parser.parse_args()
    run(args)
