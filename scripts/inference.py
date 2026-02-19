import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_INFERENCE_ARGS
from drmp.dataset.dataset import TrajectoryDataset
from drmp.model.generative_models import get_additional_inference_args, get_models
from drmp.planning.inference import create_test_subset, run_inference
from drmp.planning.inference_config import (
    ClassicalConfig,
    GenerativeModelConfig,
    MPDConfig,
    MPDSplinesConfig,
)
from drmp.utils import fix_random_seed, load_config_from_yaml, save_config_to_yaml

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print("-------- INFERENCE STARTED --------")
    print(f"algorithm: {args.algorithm}")
    if args.algorithm == "generative-model":
        print(f"model: {args.checkpoint_name}")
    print(f"dataset: {args.dataset_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_trajectories_per_task: {args.n_trajectories_per_task}")

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.algorithm == "generative-model":
            experiment_name = f"{args.checkpoint_name if args.checkpoint_name else 'generative-model'}_{timestamp}"
        elif args.algorithm == "mpd":
            experiment_name = f"mpd_{args.mpd_checkpoint_name if args.mpd_checkpoint_name else 'model'}_{timestamp}"
        else:
            experiment_name = f"{args.algorithm}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_init_config_path = os.path.join(dataset_dir, "init_config.yaml")
    dataset_init_config = load_config_from_yaml(dataset_init_config_path)

    dataset = TrajectoryDataset(**dataset_init_config, tensor_args=tensor_args)
    dataset_usage_config = {
        "apply_augmentations": False,
        "normalizer_name": "TrivialNormalizer",
        "n_control_points": args.mpd_splines_n_control_points
        if args.algorithm == "mpd-splines"
        else (
            args.rrt_grad_splines_n_control_points
            if args.algorithm == "rrt-grad-splines"
            else None
        ),
        "filtering_config": {},
    }
    if args.algorithm == "generative-model":
        dataset_usage_config = load_config_from_yaml(
            os.path.join(
                args.checkpoints_dir, args.checkpoint_name, "dataset_usage_config.yaml"
            )
        )

    train_subset, _, val_subset, _ = dataset.load_data(
        dataset_dir=dataset_dir, debug=args.debug, **dataset_usage_config
    )

    splits = eval(args.splits)

    test_subset = None
    if "test" in splits:
        test_subset = create_test_subset(
            dataset=dataset,
            n_tasks=args.n_tasks,
            threshold_start_goal_pos=args.threshold_start_goal_pos,
            use_extra_objects=args.use_extra_objects,
        )
        if test_subset is None:
            return

    model_config = None
    if args.algorithm == "generative-model":
        checkpoint_dir = os.path.join(args.checkpoints_dir, args.checkpoint_name)
        model_init_config = load_config_from_yaml(
            os.path.join(args.checkpoints_dir, args.checkpoint_name, "init_config.yaml")
        )
        model_info_config = load_config_from_yaml(
            os.path.join(args.checkpoints_dir, args.checkpoint_name, "info_config.yaml")
        )

        model_name = model_info_config["model_name"]
        model = MODELS[model_name](dataset=dataset, **model_init_config).to(device)

        model.load_state_dict(
            torch.load(
                os.path.join(
                    checkpoint_dir,
                    args.checkpoint_iter
                    or (
                        "ema_model_current_state_dict.pth"
                        if model_info_config["use_ema"]
                        else "model_current_state_dict.pth"
                    ),
                ),
                map_location=tensor_args["device"],
            )
        )
        model.eval()
        model = torch.compile(model)

        additional_args = get_additional_inference_args(model_name, vars(args))
        model_config = GenerativeModelConfig(
            model=model,
            model_name=model_name,
            use_extra_objects=args.use_extra_objects,
            t_start_guide=args.t_start_guide,
            n_guide_steps=args.n_guide_steps,
            lambda_obstacles=args.lambda_obstacles,
            lambda_velocity=args.lambda_velocity,
            lambda_acceleration=args.lambda_acceleration,
            lambda_jerk=args.lambda_jerk,
            max_grad_norm=args.max_grad_norm,
            n_interpolate=args.n_interpolate,
            additional_args=additional_args,
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

        model_config = MPDSplinesConfig(
            model=model,
            start_guide_steps_fraction=args.mpd_splines_start_guide_steps_fraction,
            n_guide_steps=args.mpd_splines_n_guide_steps,
            ddim=args.mpd_splines_ddim,
            ddim_sampling_timesteps=args.mpd_splines_ddim_sampling_timesteps,
            guide_lr=args.mpd_splines_guide_lr,
            scale_grad_prior=args.mpd_splines_scale_grad_prior,
            sigma_collision=args.mpd_splines_sigma_collision,
            sigma_gp=args.mpd_splines_sigma_gp,
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

        model_config = MPDConfig(
            model=model,
            use_extra_objects=args.use_extra_objects,
            sigma_collision=args.mpd_sigma_collision,
            sigma_gp=args.mpd_sigma_gp,
            max_grad_norm=args.mpd_max_grad_norm,
            n_interpolate=args.mpd_n_interpolate,
            start_guide_steps_fraction=args.mpd_start_guide_steps_fraction,
            n_guide_steps=args.mpd_n_guide_steps,
            ddim=args.mpd_ddim,
        )

    elif args.algorithm in [
        "rrt",
        "gpmp2",
        "grad",
        "rrt-gpmp2",
        "rrt-grad",
        "rrt-grad-splines",
    ]:
        model_config = ClassicalConfig(
            use_extra_objects=args.use_extra_objects,
            dataset=dataset,
            sampling_based_planner_name="rrt"
            if args.algorithm in ["rrt", "rrt-grad", "rrt-grad-splines", "rrt-gpmp2"]
            else None,
            optimization_based_planner_name="gpmp2"
            if args.algorithm in ["gpmp2", "rrt-gpmp2"]
            else (
                "grad"
                if args.algorithm in ["grad", "rrt-grad", "rrt-grad-splines"]
                else None
            ),
            n_sampling_steps=args.classical_n_sampling_steps,
            n_optimization_steps=args.classical_n_optimization_steps,
            n_dim=args.classical_n_dof,
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
            grad_lambda_velocity=args.grad_lambda_velocity,
            grad_lambda_acceleration=args.grad_lambda_acceleration,
            grad_lambda_jerk=args.grad_lambda_jerk,
            grad_max_grad_norm=args.grad_max_grad_norm,
            grad_n_interpolate=args.grad_n_interpolate,
        )

    else:
        raise ValueError(
            f"Unknown algorithm: {args.algorithm}. Valid options: generative-model, mpd, mpd-splines, classical"
        )

    results_dir = os.path.join(args.generations_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    config_dict = model_config.to_dict()
    config_dict.update(vars(args))
    save_config_to_yaml(config_dict, os.path.join(results_dir, "config.yaml"))

    run_inference(
        model_config=model_config,
        dataset=dataset,
        train_subset=train_subset if "train" in splits else None,
        val_subset=val_subset if "val" in splits else None,
        test_subset=test_subset,
        results_dir=results_dir,
        n_tasks=args.n_tasks,
        n_trajectories_per_task=args.n_trajectories_per_task,
        debug=args.debug,
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {}

    for key, value in DEFAULT_INFERENCE_ARGS.items():
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
