import torch
import configargparse
from datetime import datetime

from mpd.config import DEFAULT_TRAIN_ARGS, LOSS_FNS
from mpd import trainer
from mpd.models import UNET_DIM_MULTS, TemporalUnet
from mpd.trainer import get_dataset, get_model
from mpd.utils.seed import fix_random_seed
from mpd.utils.torch_utils import get_torch_device


def experiment(args):
    fix_random_seed(args.seed)

    device = get_torch_device(device=args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.dataset_dir}_bs{args.batch_size}_lr{args.lr}_steps{args.num_train_steps}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class="TrajectoryDataset",
        include_velocity=args.include_velocity,
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        logs_dir=args.logs_dir,
        save_indices=True,
        tensor_args=tensor_args,
    )

    dataset = train_subset.dataset
    print(f"Train dataset size: {len(train_subset.dataset)}")
    print(f"Val dataset size: {len(val_subset.dataset)}")

    diffusion_configs = dict(
        variance_schedule=args.variance_schedule,
        n_diffusion_steps=args.n_diffusion_steps,
        predict_epsilon=args.predict_epsilon,
    )

    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args.unet_input_dim,
        dim_mults=UNET_DIM_MULTS[args.unet_dim_mults_option],
    )

    model = get_model(
        model_class=args.diffusion_model_class,
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs,
    )

    loss_fn = LOSS_FNS[args.loss_fn]

    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        logs_dir=args.logs_dir,
        experiment_name=experiment_name,
        lr=args.lr,
        loss_fn=loss_fn,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        max_steps=args.num_train_steps,
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
        arg_type = type(value or "")
        
        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no_{key}", dest=key, action="store_false")
        else:
            kwargs = {"type": arg_type, "default": value}
            if key in special_args:
                kwargs.update(special_args[key])
            parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()
    experiment(args)
