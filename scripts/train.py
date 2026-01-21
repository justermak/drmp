import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_TRAIN_ARGS
from drmp.datasets.dataset import TrajectoryDatasetBSpline, TrajectoryDatasetDense
from drmp.models.diffusion import get_models
from drmp.train import train
from drmp.utils.torch_utils import fix_random_seed
from drmp.utils.yaml import load_config_from_yaml, save_config_to_yaml

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print("-------- TRAINING STARTED --------")
    print(f"dataset: {args.dataset_name}")
    print(f"batch size: {args.batch_size}")
    print(f"apply augmentations: {args.apply_augmentations}")
    print(f"learning rate: {args.lr}")
    print(f"number of training steps: {args.num_train_steps}")

    if args.checkpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr}_steps{args.num_train_steps}_{timestamp}"
    else:
        checkpoint_name = args.checkpoint_name

    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_config_path = os.path.join(dataset_dir, "config.yaml")
    dataset_config = load_config_from_yaml(dataset_config_path)

    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "apply_augmentations": args.apply_augmentations,
        "env_name": dataset_config["env_name"],
        "robot_name": dataset_config["robot_name"],
        "robot_margin": dataset_config["robot_margin"],
        "generating_robot_margin": dataset_config["generating_robot_margin"],
        "n_support_points": dataset_config["n_support_points"],
        "duration": dataset_config["duration"],
        "tensor_args": tensor_args,
    }
    
    dataset = None
    if args.use_splines:
        dataset_init_config["n_control_points"] = args.n_control_points
        dataset_init_config["spline_degree"] = args.spline_degree
        dataset = TrajectoryDatasetBSpline(**dataset_init_config)
    else:
        dataset = TrajectoryDatasetDense(**dataset_init_config)
    dataset.load_data(normalizer_name=args.normalizer_name)

    filtering_config = {
        "filter_collision": {} if args.filter_collision else None,
        "filter_longest_trajectories": {"portion": args.filter_longest_portion}
        if args.filter_longest_portion is not None
        else None,
        "filter_sharpest_trajectories": {"portion": args.filter_sharpest_portion}
        if args.filter_sharpest_portion is not None
        else None,
    }
    print("\nFiltering configuration:")
    for filter_name, params in filtering_config.items():
        print(f"{filter_name}: {params}")

    train_subset, train_dataloader, val_subset, val_dataloader = (
        dataset.load_train_val_split(
            batch_size=args.batch_size,
            filtering_config=filtering_config,
        )
    )

    model = MODELS[args.diffusion_model_name](
        state_dim=dataset.robot.n_dim,
        n_support_points=dataset.n_support_points if not args.use_splines else dataset.real_n_control_points,
        unet_hidden_dim=args.hidden_dim,
        unet_dim_mults=eval(args.dim_mults),
        unet_kernel_size=args.kernel_size,
        unet_resnet_block_groups=args.resnet_block_groups,
        unet_positional_encoding=args.positional_encoding,
        unet_positional_encoding_dim=args.positional_encoding_dim,
        unet_attn_heads=args.attn_heads,
        unet_attn_head_dim=args.attn_head_dim,
        unet_context_dim=args.context_dim,
        n_diffusion_steps=args.n_diffusion_steps,
        predict_epsilon=args.predict_epsilon,
    ).to(device)

    # you can load a checkpoint here

    checkpoint_dir = os.path.join(args.checkpoints_dir, "checkpoints", checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    filtering_config_path = os.path.join(checkpoint_dir, "filtering_config.yaml")
    save_config_to_yaml(filtering_config, filtering_config_path)
    print(f"\nSaved filtering config to {filtering_config_path}")

    train(
        checkpoint_name=checkpoint_name,
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        checkpoints_dir=args.checkpoints_dir,
        lr=args.lr,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        num_train_steps=args.num_train_steps,
        clip_grad=args.clip_grad,
        clip_grad_max_norm=args.clip_grad_max_norm,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_warmup=args.ema_warmup,
        ema_update_interval=args.ema_update_interval,
        use_amp=args.use_amp,
        debug=args.debug,
        tensor_args=tensor_args,
        guide_sigma_collision=args.guide_sigma_collision,
        guide_sigma_gp=args.guide_sigma_gp,
        guide_do_clip_grad=args.guide_do_clip_grad,
        guide_max_grad_norm=args.guide_max_grad_norm,
        guide_n_interpolate=args.guide_n_interpolate,
        guide_start_guide_steps_fraction=args.guide_start_guide_steps_fraction,
        guide_n_guide_steps=args.guide_n_guide_steps,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {}

    for key, value in DEFAULT_TRAIN_ARGS.items():
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
