import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_INFERENCE_LEGACY_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference.runner import run_inference, create_test_subset
from drmp.inference.runner_config import LegacyDiffusionRunnerConfig
from drmp.utils.torch_utils import fix_random_seed
from drmp.utils.yaml import load_config_from_yaml


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}
    
    print("-------- LEGACY MODEL INFERENCE STARTED --------")
    print(f"checkpoint: {args.checkpoint_name}")
    print(f"dataset: {args.dataset_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_samples: {args.n_samples}")
    
    if args.checkpoint_name is None:
        checkpoint_folders = os.listdir(args.checkpoints_dir)
        checkpoint_folders.sort(
            key=lambda x: os.path.getmtime(os.path.join(args.checkpoints_dir, x)), 
            reverse=True
        )
        args.checkpoint_name = checkpoint_folders[0]
    
    checkpoint_dir = os.path.join(args.checkpoints_dir, args.checkpoint_name)
    
    print(f"Loading legacy model from: {checkpoint_dir}")
    
    if os.path.exists(checkpoint_dir):
        model = torch.load(checkpoint_dir, map_location=device, weights_only=False)
        
    model.eval()
    
    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_config_path = os.path.join(dataset_dir, "config.yaml")
    dataset_config = load_config_from_yaml(dataset_config_path)
    dataset_config["cutoff_margin"] = args.override_cutoff_margin if args.override_cutoff_margin is not None else dataset_config["cutoff_margin"]
    
    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "env_name": dataset_config["env_name"],
        "normalizer_name": "TrivialNormalizer",
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
    
    test_subset = None
    if "test" in splits:
        test_subset = create_test_subset(
            dataset=dataset,
            n_tasks=args.n_tasks,
            threshold_start_goal_pos=args.threshold_start_goal_pos,
            tensor_args=tensor_args,
        )
    
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"legacy_{args.checkpoint_name}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    runner_config = LegacyDiffusionRunnerConfig(
        model=model,
        use_extra_objects=args.use_extra_objects,
        sigma_collision=args.sigma_collision,
        sigma_gp=args.sigma_gp,
        do_clip_grad=args.do_clip_grad,
        max_grad_norm=args.max_grad_norm,
        n_interpolate=args.n_interpolate,
        start_guide_steps_fraction=args.start_guide_steps_fraction,
        n_guide_steps=args.n_guide_steps,
        ddim=args.ddim,
    )
    
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
    
    for key, value in DEFAULT_INFERENCE_LEGACY_ARGS.items():
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
