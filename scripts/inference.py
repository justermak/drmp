import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_INFERENCE_ARGS
from drmp.datasets.dataset import TrajectoryDatasetBSpline, TrajectoryDatasetDense
from drmp.inference.runner import create_test_subset, run_inference
from drmp.inference.runner_config import (
    DiffusionRunnerConfig,
    GPMP2RRTPriorRunnerConfig,
    GPMP2UninformativeRunnerConfig,
    MPDRunnerConfig,
    MPDSplinesRunnerConfig,
    RRTConnectRunnerConfig,
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

    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "env_name": dataset_config["env_name"],
        "robot_name": dataset_config["robot_name"],
        "robot_margin": dataset_config["robot_margin"],
        "generating_robot_margin": dataset_config["generating_robot_margin"],
        "n_support_points": dataset_config["n_support_points"],
        "duration": dataset_config["duration"],
        "apply_augmentations": False,
        "tensor_args": tensor_args,
    }

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.algorithm == "diffusion":
            experiment_name = f"{args.checkpoint_name if args.checkpoint_name else 'diffusion'}_{timestamp}"
        elif args.algorithm == "mpd":
            experiment_name = f"mpd_{args.mpd_checkpoint_name if args.mpd_checkpoint_name else 'model'}_{timestamp}"
        else:
            experiment_name = f"{args.algorithm}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    if args.algorithm == "diffusion":
        if args.checkpoint_name is None:
            checkpoint_folders = os.listdir(args.checkpoints_dir)
            checkpoint_folders.sort(
                key=lambda x: os.path.getmtime(os.path.join(args.checkpoints_dir, x)),
                reverse=True,
            )
            args.checkpoint_name = checkpoint_folders[0]
        checkpoint_dir = os.path.join(args.checkpoints_dir, args.checkpoint_name)
        model_config_path = os.path.join(checkpoint_dir, "config.yaml")
        model_config = load_config_from_yaml(model_config_path)

        normalizer_name = model_config.get("normalizer_name", "TrivialNormalizer")
        use_splines = model_config.get("use_splines", False)
        if use_splines:
            dataset_init_config["n_control_points"] = model_config["n_control_points"]
            dataset_init_config["spline_degree"] = model_config["spline_degree"]
            dataset = TrajectoryDatasetBSpline(**dataset_init_config)
        else:
            dataset = TrajectoryDatasetDense(**dataset_init_config)
        dataset.load_data(normalizer_name=normalizer_name)
    elif args.algorithm == "mpd-splines":
        dataset_init_config["n_control_points"] = args.mpd_splines_n_control_points
        dataset_init_config["spline_degree"] = args.mpd_splines_spline_degree
        dataset = TrajectoryDatasetBSpline(**dataset_init_config)
        dataset.load_data()
    else:
        dataset = TrajectoryDatasetDense(**dataset_init_config)
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

    if args.algorithm == "diffusion":
        model = MODELS[model_config["model_name"]](
            state_dim=model_config["state_dim"],
            n_support_points=model_config["n_support_points"]
            if model_config["model_name"] == "GaussianDiffusion"
            else model_config["real_n_control_points"],
            unet_hidden_dim=model_config["unet_hidden_dim"],
            unet_dim_mults=eval(model_config["unet_dim_mults"]),
            unet_kernel_size=model_config["unet_kernel_size"],
            unet_resnet_block_groups=model_config["unet_resnet_block_groups"],
            unet_positional_encoding=model_config["unet_positional_encoding"],
            unet_positional_encoding_dim=model_config["unet_positional_encoding_dim"],
            unet_attn_heads=model_config["unet_attn_heads"],
            unet_attn_head_dim=model_config["unet_attn_head_dim"],
            unet_context_dim=model_config["unet_context_dim"],
            n_diffusion_steps=model_config["n_diffusion_steps"],
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

    elif args.algorithm == "mpd-splines":
        checkpoint_path = os.path.join(
            args.mpd_splines_checkpoints_dir, args.mpd_splines_checkpoint_name
        )

        print(f"Loading MPD-Splines model from: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(
                f"MPD-Splines model checkpoint not found: {checkpoint_path}"
            )

        model.eval()
        model = torch.compile(model)

        runner_config = MPDSplinesRunnerConfig(
            model=model,
            start_guide_steps_fraction=args.mpd_splines_start_guide_steps_fraction,
            n_guide_steps=args.mpd_splines_n_guide_steps,
            ddim=args.mpd_splines_ddim,
            ddim_sampling_timesteps=args.mpd_splines_ddim_sampling_timesteps,
            guide_lr=args.mpd_splines_guide_lr,
            scale_grad_prior=args.mpd_splines_scale_grad_prior,
            sigma_collision=args.mpd_splines_sigma_collision,
            sigma_gp=args.mpd_splines_sigma_gp,
            do_clip_grad=args.mpd_splines_do_clip_grad,
            max_grad_norm=args.mpd_splines_max_grad_norm,
            n_interpolate=args.mpd_splines_n_interpolate,
            use_extra_objects=args.use_extra_objects,
        )

    elif args.algorithm == "mpd":
        checkpoint_path = os.path.join(
            args.mpd_checkpoints_dir, args.mpd_checkpoint_name
        )

        print(f"Loading MPD model from: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            model = torch.load(checkpoint_path, map_location=device, weights_only=False)
        else:
            raise FileNotFoundError(
                f"MPD model checkpoint not found: {checkpoint_path}"
            )

        model.eval()
        model = torch.compile(model)

        runner_config = MPDRunnerConfig(
            model=model,
            use_extra_objects=args.use_extra_objects,
            sigma_collision=args.mpd_sigma_collision,
            sigma_gp=args.mpd_sigma_gp,
            do_clip_grad=args.mpd_do_clip_grad,
            max_grad_norm=args.mpd_max_grad_norm,
            n_interpolate=args.mpd_n_interpolate,
            start_guide_steps_fraction=args.mpd_start_guide_steps_fraction,
            n_guide_steps=args.mpd_n_guide_steps,
            ddim=args.mpd_ddim,
        )

    elif args.algorithm == "rrt-connect":
        runner_config = RRTConnectRunnerConfig(
            use_extra_objects=args.use_extra_objects,
            sample_steps=args.classical_sample_steps,
            rrt_connect_max_step_size=args.rrt_connect_max_step_size,
            rrt_connect_max_radius=args.rrt_connect_max_radius,
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
            n_dof=args.classical_n_dof,
            rrt_connect_max_step_size=args.rrt_connect_max_step_size,
            rrt_connect_max_radius=args.rrt_connect_max_radius,
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
        raise ValueError(
            f"Unknown algorithm: {args.algorithm}. Valid options: diffusion, mpd, rrt-connect, gpmp2-uninformative, gpmp2-rrt-prior"
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
