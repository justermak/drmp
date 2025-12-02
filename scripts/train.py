import os
from datetime import datetime

import configargparse
import torch

from drmp.config import DEFAULT_TRAIN_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.models.models import get_models
from drmp.train import train
from drmp.utils.seed import fix_random_seed
from drmp.utils.yaml import load_config_from_yaml

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    print(f"-------- TRAINING STARTED --------")
    print(f"dataset: {args.dataset_name}")
    print(f"batch size: {args.batch_size}")
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

    # Filter config to only include parameters accepted by TrajectoryDataset.__init__
    # Override paths with command line arguments to ensure correct paths
    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "normalizer_name": dataset_config["normalizer_name"],
        "obstacle_cutoff_margin": dataset_config["obstacle_cutoff_margin"],
        "env_name": dataset_config["env_name"],
        "tensor_args": tensor_args,
    }

    dataset = TrajectoryDataset(**dataset_init_config)
    dataset.load_data()
    train_subset, train_dataloader, val_subset, val_dataloader = (
        dataset.load_train_val_split(batch_size=args.batch_size)
    )

    print(f"Train dataset size: {len(train_subset.dataset)}")
    print(f"Val dataset size: {len(val_subset.dataset)}")

    model = MODELS[args.diffusion_model_name](
        state_dim=train_subset.dataset.state_dim,
        n_support_points=train_subset.dataset.n_support_points,
        unet_input_dim=args.unet_input_dim,
        unet_dim_mults=eval(args.unet_dim_mults),
        time_emb_dim=args.time_emb_dim,
        self_attention=args.self_attention,
        conditioning_embed_dim=args.conditioning_embed_dim,
        conditioning_type=args.conditioning_type,
        attention_num_heads=args.attention_num_heads,
        attention_dim_head=args.attention_dim_head,
        variance_schedule=args.variance_schedule,
        n_diffusion_steps=args.n_diffusion_steps,
        predict_epsilon=args.predict_epsilon,
    ).to(device)
    
    # you can load a checkpoint here

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
        early_stopper_patience=args.early_stopper_patience,
        steps_per_validation=args.steps_per_validation,
        use_amp=args.use_amp,
        debug=args.debug,
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {
        "variance_schedule": {"choices": ["exponential", "cosine"]},
    }

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
