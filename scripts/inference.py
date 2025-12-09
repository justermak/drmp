import os
from copy import copy
from datetime import datetime

import configargparse
import torch
from einops._torch_specific import allow_ops_in_compiled_graph
from torch.utils.data import Subset

from drmp.config import DEFAULT_INFERENCE_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference import run_inference
from drmp.models.diffusion import get_models
from drmp.utils.torch_utils import fix_random_seed, freeze_torch_model_params
from drmp.utils.yaml import load_config_from_yaml

allow_ops_in_compiled_graph()

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print(f"-------- INFERENCE STARTED --------")
    print(f"model: {args.checkpoint_name}")
    print(f"dataset: {args.dataset_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_samples: {args.n_samples}")

    if args.checkpoint_name is None:
        checkpoint_folders = os.listdir(args.checkpoints_dir)
        checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(args.checkpoints_dir, x)), reverse=True)
        args.checkpoint_name = checkpoint_folders[0]
    checkpoint_dir = os.path.join(args.checkpoints_dir, args.checkpoint_name)
    model_config_path = os.path.join(checkpoint_dir, "config.yaml")
    model_config = load_config_from_yaml(model_config_path)

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

    test_subset = None

    if "test" in splits:

        start_pos, goal_pos, success = dataset.env.random_collision_free_start_goal(
            robot=dataset.robot,
            n_samples=args.n_tasks,
            threshold_start_goal_pos=args.threshold_start_goal_pos,
        )
        if not success:
            print(
                "Could not find sufficient collision-free start/goal pairs for test tasks, try reducing the threshold, robot margin or object density"
            )
            return
        test_dataset = copy(dataset)
        test_dataset.n_trajs = args.n_tasks
        test_dataset.trajs_normalized = torch.empty((args.n_tasks,), **tensor_args)
        test_dataset.start_states = torch.cat(
            [start_pos, torch.zeros_like(start_pos)], dim=-1
        )
        test_dataset.goal_states = torch.cat(
            [goal_pos, torch.zeros_like(goal_pos)], dim=-1
        )
        test_dataset.start_states_normalized = test_dataset.normalizer.normalize(
            test_dataset.start_states
        )
        test_dataset.goal_states_normalized = test_dataset.normalizer.normalize(
            test_dataset.goal_states
        )
        test_subset = Subset(test_dataset, list(range(args.n_tasks)))

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.checkpoint_name}_{timestamp}"
    else:
        experiment_name = args.experiment_name

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
    freeze_torch_model_params(model)
    model = torch.compile(model)

    run_inference(
        model=model,
        dataset=dataset,
        train_subset=train_subset if "train" in splits else None,
        val_subset=val_subset if "val" in splits else None,
        test_subset=test_subset,
        generations_dir=args.generations_dir,
        experiment_name=experiment_name,
        use_extra_objects=args.use_extra_objects,
        sigma_collision=args.sigma_collision,
        sigma_gp=args.sigma_gp,
        do_clip_grad=args.do_clip_grad,
        max_grad_norm=args.max_grad_norm,
        n_interpolate=args.n_interpolate,
        start_guide_steps_fraction=args.start_guide_steps_fraction,
        n_tasks=args.n_tasks,
        threshold_start_goal_pos=args.threshold_start_goal_pos,
        n_samples=args.n_samples,
        n_guide_steps=args.n_guide_steps,
        ddim=args.ddim,
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
