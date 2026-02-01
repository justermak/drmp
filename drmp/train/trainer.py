import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from drmp.datasets.dataset import TrajectoryDatasetBase, TrajectoryDatasetBSpline
from drmp.models.diffusion import DiffusionModelBase
from drmp.planning.costs import (
    CostCollision,
    CostComposite,
    CostGPTrajectory,
    CostJointAcceleration,
    CostJointPosition,
    CostJointVelocity,
    CostObstacles,
)
from drmp.planning.guide import Guide
from drmp.train.logs import log
from drmp.utils.torch_timer import TimerCUDA
from drmp.utils.yaml import save_config_to_yaml


class EMA:
    def __init__(self, beta: float = 0.995) -> None:
        super().__init__()
        self.beta: float = beta

    def update_model_average(
        self, ema_model: torch.nn.Module, current_model: torch.nn.Module
    ) -> None:
        for ema_params, current_params in zip(
            ema_model.parameters(), current_model.parameters()
        ):
            old_weight, current_weight = ema_params.data, current_params.data
            ema_params.data = old_weight * self.beta + (1 - self.beta) * current_weight


def train_step(
    model: DiffusionModelBase,
    train_batch_dict: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    clip_grad: Union[bool, float],
    clip_grad_max_norm: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        train_losses = model.compute_loss(train_batch_dict)

    train_loss = sum(train_losses.values())
    train_losses_log = {k: v.mean().item() for k, v in train_losses.items()}

    optimizer.zero_grad()
    scaler.scale(train_loss).backward()

    if clip_grad:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_grad_max_norm if isinstance(clip_grad, bool) else clip_grad,
        )

    scaler.step(optimizer)
    scaler.update()

    return train_loss, train_losses_log


def val_step(
    model: DiffusionModelBase,
    val_dataloader: DataLoader,
) -> Tuple[float, Dict[str, float]]:
    val_losses = defaultdict(list)
    val_loss_list = []

    for step_val, batch_dict_val in enumerate(val_dataloader):
        val_losses = model.compute_loss(batch_dict_val)
        if step_val == 0:
            val_losses_log = {k: [] for k in val_losses.keys()}

        val_loss = sum(val_losses.values())
        val_loss_list.append(val_loss.item())
        for k, v in val_losses.items():
            val_losses_log[k].append(v.item())

    val_loss = np.mean(val_loss_list)
    val_losses_log = {k: np.mean(v) for k, v in val_losses_log.items()}

    return val_loss, val_losses_log


def save_model_to_disk(
    model: DiffusionModelBase,
    epoch: int,
    step: int,
    checkpoint_dir: Optional[str] = None,
    prefix: str = "",
) -> None:
    if model is None:
        return

    if hasattr(model, "is_frozen") and model.is_frozen:
        return

    torch.save(
        model.state_dict(),
        os.path.join(checkpoint_dir, f"{prefix}_current_state_dict.pth"),
    )
    torch.save(
        model.state_dict(),
        os.path.join(
            checkpoint_dir, f"{prefix}_epoch_{epoch:04d}_iter_{step:06d}_state_dict.pth"
        ),
    )
    torch.save(
        model,
        os.path.join(checkpoint_dir, f"{prefix}_current.pth"),
    )
    torch.save(
        model,
        os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch:04d}_iter_{step:06d}.pth"),
    )


def save_losses_to_disk(
    train_losses: List[Tuple[int, Dict[str, float]]],
    val_losses: List[Tuple[int, Dict[str, float]]],
    checkpoint_dir: Optional[str] = None,
) -> None:
    np.save(os.path.join(checkpoint_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(checkpoint_dir, "val_losses.npy"), val_losses)


def end_training(
    model: DiffusionModelBase,
    ema_model: DiffusionModelBase,
    ema: EMA,
    epoch: int,
    step: int,
    ema_warmup: int,
    train_losses: List[Tuple[int, Dict[str, float]]],
    val_losses: List[Tuple[int, Dict[str, float]]],
    checkpoint_dir: str,
    tensorboard_writer: SummaryWriter,
) -> None:
    if ema_model is not None:
        if step < ema_warmup:
            ema_model.load_state_dict(model.state_dict())
        ema.update_model_average(ema_model, model)

    save_model_to_disk(
        model=model,
        epoch=epoch,
        step=step,
        checkpoint_dir=checkpoint_dir,
        prefix="model",
    )
    if ema_model is not None:
        save_model_to_disk(
            model=ema_model,
            epoch=epoch,
            step=step,
            checkpoint_dir=checkpoint_dir,
            prefix="ema_model",
        )
    save_losses_to_disk(
        train_losses=train_losses, val_losses=val_losses, checkpoint_dir=checkpoint_dir
    )

    tensorboard_writer.close()


def train(
    model: DiffusionModelBase,
    train_dataloader: DataLoader,
    train_subset: Subset,
    val_dataloader: DataLoader,
    val_subset: Subset,
    lr: float,
    num_train_steps: int,
    checkpoint_name: Optional[str],
    checkpoints_dir: str,
    log_interval: int,
    checkpoint_interval: int,
    clip_grad: Union[bool, float],
    clip_grad_max_norm: float,
    use_ema: bool,
    ema_decay: float,
    ema_warmup: int,
    ema_update_interval: int,
    use_amp: bool,
    ddim: bool,
    shortcut_steps: int,
    n_guide_steps: int,
    t_start_guide: int,
    guide_lambda_obstacles: float,
    guide_lambda_position: float,
    guide_lambda_velocity: float,
    guide_lambda_acceleration: float,
    guide_max_grad_norm: float,
    guide_n_interpolate: int,
    debug: bool,
    tensor_args: Dict[str, Any],
) -> None:
    epochs = int(np.ceil(num_train_steps / len(train_dataloader)))
    step = 0

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    scaler = torch.amp.GradScaler(device=tensor_args["device"], enabled=use_amp)

    os.makedirs(checkpoints_dir, exist_ok=True)
    if checkpoint_name:
        stats_dir = os.path.join(checkpoints_dir, "stats", checkpoint_name)
        checkpoint_dir = os.path.join(checkpoints_dir, "checkpoints", checkpoint_name)
    else:
        stats_dir = os.path.join(checkpoints_dir, "stats")
        checkpoint_dir = os.path.join(checkpoints_dir, "checkpoints")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=stats_dir)

    ema = None
    ema_model = None
    if use_ema:
        ema = EMA(beta=ema_decay)
        ema_model = copy.deepcopy(model)

    dataset: TrajectoryDatasetBase = train_subset.dataset

    collision_cost = None
    if guide_lambda_obstacles is not None:
        collision_cost = CostObstacles(
            robot=dataset.robot,
            env=dataset.env,
            n_support_points=dataset.n_support_points,
            lambda_obstacles=guide_lambda_obstacles,
            use_extra_objects=False,
            tensor_args=tensor_args,
        )

        collision_cost_extra = CostObstacles(
            robot=dataset.robot,
            env=dataset.env,
            n_support_points=dataset.n_support_points,
            lambda_obstacles=guide_lambda_obstacles,
            use_extra_objects=True,
            tensor_args=tensor_args,
        )
        
    position_cost = None
    if guide_lambda_position is not None:
        position_cost = CostJointPosition(
            robot=dataset.robot,
            n_support_points=dataset.n_support_points,
            lambda_position=guide_lambda_position,
            tensor_args=tensor_args,
        )

    velocity_cost = None
    if guide_lambda_velocity is not None:
        velocity_cost = CostJointVelocity(
            robot=dataset.robot,
            n_support_points=dataset.n_support_points,
            lambda_velocity=guide_lambda_velocity,
            tensor_args=tensor_args,
        )

    acceleration_cost = None
    if guide_lambda_acceleration is not None:
        acceleration_cost = CostJointAcceleration(
            robot=dataset.robot,
            n_support_points=dataset.n_support_points,
            lambda_acceleration=guide_lambda_acceleration,
            tensor_args=tensor_args,
        )
    
    costs = [cost for cost in [collision_cost, position_cost, velocity_cost, acceleration_cost] if cost is not None]
    costs_extra = [cost for cost in [collision_cost_extra, position_cost, velocity_cost, acceleration_cost] if cost is not None]

    guide = Guide(
        dataset=dataset,
        costs=costs,
        max_grad_norm=guide_max_grad_norm,
        n_interpolate=guide_n_interpolate,
    )

    guide_extra = Guide(
        dataset=dataset,
        costs=costs_extra,
        max_grad_norm=guide_max_grad_norm,
        n_interpolate=guide_n_interpolate,
    )

    config = {
        "checkpoints_dir": checkpoints_dir,
        "checkpoint_name": checkpoint_name,
        "checkpoint_dir": checkpoint_dir,
        "lr": lr,
        "batch_size": train_dataloader.batch_size,
        "model_name": model.__class__.__name__,
        "state_dim": model.state_dim,
        "horizon": model.horizon,
        "unet_hidden_dim": model.unet_hidden_dim,
        "unet_dim_mults": str(model.unet_dim_mults),
        "unet_kernel_size": model.unet_kernel_size,
        "unet_resnet_block_groups": model.unet_resnet_block_groups,
        "unet_positional_encoding": model.unet_positional_encoding,
        "unet_positional_encoding_dim": model.unet_positional_encoding_dim,
        "unet_attn_heads": model.unet_attn_heads,
        "unet_attn_head_dim": model.unet_attn_head_dim,
        "unet_context_dim": model.unet_context_dim,
        "n_diffusion_steps": model.n_diffusion_steps,
        "predict_epsilon": model.predict_epsilon,
        "log_interval": log_interval,
        "checkpoint_interval": checkpoint_interval,
        "clip_grad": clip_grad,
        "clip_grad_max_norm": clip_grad_max_norm,
        "use_ema": use_ema,
        "ema_decay": ema_decay,
        "ema_warmup": ema_warmup,
        "ema_update_interval": ema_update_interval,
        "use_amp": use_amp,
        "ddim": ddim,
        "shortcut_steps": shortcut_steps,
        "n_guide_steps": n_guide_steps,
        "t_start_guide": t_start_guide,
        "guide_lambda_obstacles": guide_lambda_obstacles,
        "guide_lambda_position": guide_lambda_position,
        "guide_lambda_velocity": guide_lambda_velocity,
        "guide_lambda_acceleration": guide_lambda_acceleration,
        "guide_max_grad_norm": guide_max_grad_norm,
        "guide_n_interpolate": guide_n_interpolate,
        "debug": debug,
    }

    save_config_to_yaml(config, os.path.join(checkpoint_dir, "config.yaml"))
    save_model_to_disk(
        model=model, epoch=0, step=0, checkpoint_dir=checkpoint_dir, prefix="model"
    )
    if ema_model is not None:
        save_model_to_disk(
            model=ema_model,
            epoch=0,
            step=0,
            checkpoint_dir=checkpoint_dir,
            prefix="ema_model",
        )

    try:
        with tqdm(
            total=len(train_dataloader) * epochs,
            mininterval=1 if debug else 60,
        ) as pbar:
            model.train()
            train_losses = []
            val_losses = []
            for epoch in range(epochs):
                for i, train_batch_dict in enumerate(train_dataloader):
                    with TimerCUDA() as t_training_loss:
                        model.train()
                        train_loss, train_losses_log = train_step(
                            model=model,
                            train_batch_dict=train_batch_dict,
                            optimizer=optimizer,
                            scaler=scaler,
                            use_amp=use_amp,
                            clip_grad=clip_grad,
                            clip_grad_max_norm=clip_grad_max_norm,
                        )

                    if ema_model is not None:
                        if step % ema_update_interval == 0:
                            if step < ema_warmup:
                                ema_model.load_state_dict(model.state_dict())
                            ema.update_model_average(ema_model, model)

                    if step % log_interval == 0:
                        print("=" * 80)
                        print(f"step: {step}")
                        print(f"t_train_loss: {t_training_loss.elapsed:.4f} sec")
                        print(f"Total train loss {train_loss:.4f}")
                        print(f"Train losses {train_losses_log}")

                        train_losses.append((step, train_losses_log))

                        with TimerCUDA() as t_train_summary:
                            log(
                                step=step,
                                model=ema_model if ema_model is not None else model,
                                subset=train_subset,
                                train_losses=train_losses_log,
                                prefix="TRAIN ",
                                debug=debug,
                                tensorboard_writer=tensorboard_writer,
                                ddim=ddim,
                                shortcut_steps=shortcut_steps,
                                guide=guide,
                                guide_extra=guide_extra,
                                t_start_guide=t_start_guide,
                                n_guide_steps=n_guide_steps,
                            )
                        print(f"t_train_summary: {t_train_summary.elapsed:.4f} sec")

                        validation_losses_log = {}
                        with TimerCUDA() as t_validation_loss:
                            total_val_loss, validation_losses_log = val_step(
                                model=model,
                                val_dataloader=val_dataloader,
                            )

                        print(f"t_val_loss: {t_validation_loss.elapsed:.4f} sec")
                        print(f"Total val loss {total_val_loss:.4f}")
                        print(f"Val losses {validation_losses_log}")

                        val_losses.append((step, validation_losses_log))

                        with TimerCUDA() as t_val_summary:
                            log(
                                step=step,
                                model=ema_model if ema_model is not None else model,
                                subset=val_subset,
                                val_losses=validation_losses_log,
                                prefix="VAL ",
                                debug=debug,
                                tensorboard_writer=tensorboard_writer,
                                ddim=ddim,
                                shortcut_steps=shortcut_steps,
                                guide=guide,
                                guide_extra=guide_extra,
                                t_start_guide=t_start_guide,
                                n_guide_steps=n_guide_steps,
                            )
                        print(f"t_val_summary: {t_val_summary.elapsed:.4f} sec")

                    pbar.update(1)
                    step += 1

                    if (checkpoint_interval is not None) and (
                        step % checkpoint_interval == 0
                    ):
                        save_model_to_disk(
                            model=model,
                            epoch=epoch,
                            step=step,
                            checkpoint_dir=checkpoint_dir,
                            prefix="model",
                        )
                        if ema_model is not None:
                            save_model_to_disk(
                                model=ema_model,
                                epoch=epoch,
                                step=step,
                                checkpoint_dir=checkpoint_dir,
                                prefix="ema_model",
                            )
                        save_losses_to_disk(
                            train_losses=train_losses,
                            val_losses=val_losses,
                            checkpoint_dir=checkpoint_dir,
                        )

            end_training(
                model=model,
                ema_model=ema_model,
                ema=ema,
                epoch=epoch,
                step=step,
                ema_warmup=ema_warmup,
                train_losses=train_losses,
                val_losses=val_losses,
                checkpoint_dir=checkpoint_dir,
                tensorboard_writer=tensorboard_writer,
            )

            print("\n-------- TRAINING FINISHED --------")

    except KeyboardInterrupt:
        print("\n\n-------- TRAINING INTERRUPTED --------")
        print(f"Saving checkpoint at step {step}...")

        end_training(
            model=model,
            ema_model=ema_model,
            ema=ema,
            epoch=epoch,
            step=step,
            ema_warmup=ema_warmup,
            train_losses=train_losses,
            val_losses=val_losses,
            checkpoint_dir=checkpoint_dir,
            tensorboard_writer=tensorboard_writer,
        )

        print("-------- TRAINING STOPPED --------\n")
        raise
