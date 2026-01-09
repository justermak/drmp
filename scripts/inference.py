import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_INFERENCE_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference.runner import run_inference, create_test_subset
from drmp.inference.runner_config import (
    DiffusionRunnerConfig,
    LegacyDiffusionRunnerConfig,
    RRTConnectRunnerConfig,
    GPMP2UninformativeRunnerConfig,
    GPMP2RRTPriorRunnerConfig,
)
from drmp.models.diffusion import get_models
from drmp.utils.torch_utils import fix_random_seed
from drmp.utils.yaml import load_config_from_yaml

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print("-------- INFERENCE STARTED --------")
    print(f"algorithm: {args.algorithm}")
    if args.algorithm == "diffusion":
        print(f"model: {args.checkpoint_name}")
    print(f"dataset: {args.dataset_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_samples: {args.n_samples}")

    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_config_path = os.path.join(dataset_dir, "config.yaml")
    dataset_config = load_config_from_yaml(dataset_config_path)
    normalizer_name = "TrivialNormalizer" if args.algorithm == "legacy-diffusion" else dataset_config["normalizer_name"]

    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "env_name": dataset_config["env_name"],
        "normalizer_name": normalizer_name,
        "robot_margin": dataset_config["robot_margin"],
        "generating_robot_margin": dataset_config["generating_robot_margin"],
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
            use_extra_objects=args.use_extra_objects,
            tensor_args=tensor_args,
        )
        if test_subset is None:
            return
        
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.algorithm == "diffusion":
            experiment_name = f"{args.checkpoint_name if args.checkpoint_name else 'diffusion'}_{timestamp}"
        elif args.algorithm == "legacy-diffusion":
            experiment_name = f"legacy_{args.legacy_checkpoint_name if args.legacy_checkpoint_name else 'model'}_{timestamp}"
        else:
            experiment_name = f"{args.algorithm}_{timestamp}"
    else:
        experiment_name = args.experiment_name
        
        
    if args.algorithm == "diffusion": 
        if args.checkpoint_name is None:
            checkpoint_folders = os.listdir(args.checkpoints_dir)
            checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoints_dir, x)), reverse=True)
            args.checkpoint_name = checkpoint_folders[0]
        checkpoint_dir = os.path.join(args.checkpoints_dir, args.checkpoint_name)
        model_config_path = os.path.join(checkpoint_dir, "config.yaml")
        model_config = load_config_from_yaml(model_config_path)
         
        model = MODELS[model_config["model_name"]](
            state_dim=model_config["state_dim"],
            n_support_points=model_config["n_support_points"],
            unet_hidden_dim=model_config["unet_hidden_dim"],
            unet_dim_mults=eval(model_config["unet_dim_mults"]),
            unet_kernel_size=model_config["unet_kernel_size"],
            unet_resnet_block_groups=model_config["unet_resnet_block_groups"],
            unet_random_fourier_features=model_config["unet_random_fourier_features"],
            unet_learned_sin_dim=model_config["unet_learned_sin_dim"],
            unet_attn_heads=model_config["unet_attn_heads"],
            unet_attn_head_dim=model_config["unet_attn_head_dim"],
            unet_context_dim=model_config["unet_context_dim"],
            variance_schedule=model_config["variance_schedule"],
            n_diffusion_steps=model_config["n_diffusion_steps"],
            clip_denoised=model_config["clip_denoised"],
            predict_epsilon=model_config["predict_epsilon"],
        ).to(device)

        model.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_dir,
                    "ema_model_current_state_dict.pth"
                    if model_config["use_ema"]
                    else "model_current_state_dict.pth",
                ),
                map_location=tensor_args["device"],
            )
        )
        model.eval()
        model = torch.compile(model)

        runner_config = DiffusionRunnerConfig(
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
        
    elif args.algorithm == "legacy-diffusion":
        checkpoint_path = os.path.join(args.legacy_checkpoints_dir, args.legacy_checkpoint_name)
        
        print(f"Loading legacy model from: {checkpoint_path}")
        
        if os.path.exists(checkpoint_path):
            model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(f"Legacy model checkpoint not found: {checkpoint_path}")
            
        model.eval()
        model = torch.compile(model)

        runner_config = LegacyDiffusionRunnerConfig(
            model=model,
            use_extra_objects=args.use_extra_objects,
            sigma_collision=args.legacy_sigma_collision,
            sigma_gp=args.legacy_sigma_gp,
            do_clip_grad=args.legacy_do_clip_grad,
            max_grad_norm=args.legacy_max_grad_norm,
            n_interpolate=args.legacy_n_interpolate,
            start_guide_steps_fraction=args.legacy_start_guide_steps_fraction,
            n_guide_steps=args.legacy_n_guide_steps,
            ddim=args.legacy_ddim,
        )

    elif args.algorithm == "rrt-connect":
        runner_config = RRTConnectRunnerConfig(
            use_extra_objects=args.use_extra_objects,
            sample_steps=args.classical_sample_steps,
            use_parallel=args.classical_use_parallel,
            max_processes=args.classical_max_processes,
            rrt_connect_step_size=args.rrt_connect_step_size,
            rrt_connect_n_radius=args.rrt_connect_n_radius,
            rrt_connect_n_samples=args.rrt_connect_n_samples,
            seed=args.seed,
        )

    elif args.algorithm == "gpmp2-uninformative":
        runner_config = GPMP2UninformativeRunnerConfig(
            use_extra_objects=args.use_extra_objects,
            opt_steps=args.classical_opt_steps,
            n_dof=args.classical_n_dof,
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

    elif args.algorithm == "gpmp2-rrt-prior":
        runner_config = GPMP2RRTPriorRunnerConfig(
            use_extra_objects=args.use_extra_objects,
            sample_steps=args.classical_sample_steps,
            opt_steps=args.classical_opt_steps,
            use_parallel=args.classical_use_parallel,
            max_processes=args.classical_max_processes,
            n_dof=args.classical_n_dof,
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
        raise ValueError(f"Unknown algorithm: {args.algorithm}. Valid options: diffusion, legacy-diffusion, rrt-connect, gpmp2-uninformative, gpmp2-rrt-prior")


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

    special_args = {}

    for key, value in DEFAULT_INFERENCE_ARGS.items():
        arg_name = f"--{key}"
        arg_type = type(value if value is not None else str)

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
