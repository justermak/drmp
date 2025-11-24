import copy
from typing import Dict, List, Tuple, Optional, Any, Callable, Union

import numpy as np
import os
import torch
from collections import defaultdict
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from mpd.utils.torch_timer import TimerCUDA
from mpd.utils.torch_utils import dict_to_device
from mpd.config import DEFAULT_TENSOR_ARGS, DEFAULT_TRAIN_ARGS
from mpd.trainer.logs import log


def get_num_epochs(num_train_steps: int, batch_size: int, dataset_len: int) -> int:
    return int(np.ceil(num_train_steps * batch_size / dataset_len))


class EMA:
    beta: float

    def __init__(self, beta: float = 0.995) -> None:
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model: torch.nn.Module, current_model: torch.nn.Module) -> None:
        for ema_params, current_params in zip(
            ema_model.parameters(), current_model.parameters()
        ):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def train_step(
    model: torch.nn.Module,
    train_batch_dict: Dict[str, Any],
    loss_fn: Callable,
    train_subset: Subset,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    clip_grad: Union[bool, float],
    clip_grad_max_norm: float,
    ema_model: Optional[torch.nn.Module],
    ema: Optional[EMA],
    step: int,
    ema_warmup: int,
    ema_update_interval: int,
    tensor_args: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Execute a single training step."""

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        train_losses = loss_fn(model, train_batch_dict, train_subset.dataset)

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

    # Update EMA
    if ema_model is not None:
        if step % ema_update_interval == 0:
            if step < ema_warmup:
                ema_model.load_state_dict(model.state_dict())
            ema.update_model_average(ema_model, model)

    return train_loss, train_losses_log


def val_step(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    loss_fn: Callable,
    val_subset: Subset,
    step: int,
    steps_per_validation: int,
    tensor_args: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Execute validation over multiple batches."""
    val_losses = defaultdict(list)
    val_loss_list = []

    for step_val, batch_dict_val in enumerate(val_dataloader):
        batch_dict_val = dict_to_device(batch_dict_val, tensor_args["device"])
        val_losses = loss_fn(
            model, batch_dict_val, val_subset.dataset, step=step
        )
        if step_val == 0:
            val_losses_log = {k: [] for k in val_losses.keys()}

        val_loss = sum(val_losses.values())
        val_loss_list.append(val_loss.item())
        for k, v in val_losses.items():
            val_losses_log[k].append(v.item())

        if step_val == steps_per_validation:
            break

    val_loss = np.mean(val_loss_list)
    val_losses_log = {f"VALIDATION {k}": np.mean(v) for k, v in val_losses_log.items()}

    return val_loss, val_losses_log

class EarlyStopper:
    patience: int
    min_delta: float
    counter: int
    min: Optional[float]
    early_stop_triggered: bool

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min = None
        self.early_stop_triggered = False
        
    def __call__(self, val_metric: float) -> bool:
        if self.patience == -1:
            return False
        
        if self.min is None:
            self.min = val_metric
        elif val_metric > self.min + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop_triggered = True
                return True
        else:
            self.min = val_metric
            self.counter = 0
            
        return False



def save_model_to_disk(
    model: Optional[torch.nn.Module], 
    epoch: int, 
    total_steps: int, 
    checkpoints_dir: Optional[str] = None
) -> None:
    if model is None:
        return

    if hasattr(model, "is_frozen") and model.is_frozen:
        return

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "current_state_dict.pth"))
    torch.save(
        model.state_dict(),
        os.path.join(
            checkpoints_dir,
            f"epoch_{epoch:04d}_iter_{total_steps:06d}_state_dict.pth",
        ),
    )
    torch.save(model, os.path.join(checkpoints_dir, "current.pth"))
    torch.save(
        model,
        os.path.join(
            checkpoints_dir, f"epoch_{epoch}_iter_{total_steps:06d}.pth"
        ),
    )


def save_losses_to_disk(
    train_losses: List[Tuple[int, Dict[str, float]]], 
    val_losses: List[Tuple[int, Dict[str, float]]], 
    checkpoints_dir: Optional[str] = None
) -> None:
    np.save(os.path.join(checkpoints_dir, f"train_losses.npy"), train_losses)
    np.save(os.path.join(checkpoints_dir, f"val_losses.npy"), val_losses)


def end_training(
    model: torch.nn.Module, 
    ema_model: Optional[torch.nn.Module], 
    ema: Optional[EMA], 
    epoch: int, 
    step: int, 
    ema_warmup: int, 
    train_losses_l: List[Tuple[int, Dict[str, float]]], 
    validation_losses_l: List[Tuple[int, Dict[str, float]]], 
    checkpoints_dir: str, 
    tensorboard_writer: SummaryWriter
) -> None:
    if ema_model is not None:
        if step < ema_warmup:
            ema_model.load_state_dict(model.state_dict())
        ema.update_model_average(ema_model, model)

    save_model_to_disk(model, epoch, step, checkpoints_dir)
    if ema_model is not None:
        save_model_to_disk(ema_model, epoch, step, checkpoints_dir)
    save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)
    
    tensorboard_writer.close()

def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    train_subset: Subset,
    experiment_name: Optional[str] = None,
    loss_fn: Callable = DEFAULT_TRAIN_ARGS["loss_fn"],
    logs_dir: str = DEFAULT_TRAIN_ARGS["logs_dir"],
    lr: float = DEFAULT_TRAIN_ARGS["lr"],
    log_interval: int = DEFAULT_TRAIN_ARGS["log_interval"],
    checkpoint_interval: int = DEFAULT_TRAIN_ARGS["checkpoint_interval"],
    val_dataloader: Optional[DataLoader] = None,
    val_subset: Optional[Subset] = None,
    clip_grad: Union[bool, float] = DEFAULT_TRAIN_ARGS["clip_grad"],
    clip_grad_max_norm: float = DEFAULT_TRAIN_ARGS["clip_grad_max_norm"],
    optimizer: Optional[torch.optim.Optimizer] = None,
    steps_per_validation: int = DEFAULT_TRAIN_ARGS["steps_per_validation"],
    max_steps: Optional[int] = None,
    use_ema: bool = DEFAULT_TRAIN_ARGS["use_ema"],
    ema_decay: float = DEFAULT_TRAIN_ARGS["ema_decay"],
    ema_warmup: int = DEFAULT_TRAIN_ARGS["ema_warmup"],
    ema_update_interval: int = DEFAULT_TRAIN_ARGS["ema_update_interval"],
    use_amp: bool = DEFAULT_TRAIN_ARGS["use_amp"],
    early_stopper_patience: int = DEFAULT_TRAIN_ARGS["early_stopper_patience"],
    debug: bool = DEFAULT_TRAIN_ARGS["debug"],
    tensor_args: Dict[str, Any] = DEFAULT_TENSOR_ARGS,
    **kwargs: Any,
) -> None:
    print(f"\n------- TRAINING STARTED -------\n")
    
        
    epochs = get_num_epochs(max_steps, train_dataloader.batch_size, len(train_subset.dataset))
    step = 0    
    
    if optimizer is None:
        optimizer = torch.optim.Adam(lr=lr, params=model.parameters())

    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    os.makedirs(logs_dir, exist_ok=True)
    if experiment_name:
        stats_dir = os.path.join(logs_dir, "stats", experiment_name)
        checkpoints_dir = os.path.join(logs_dir, "checkpoints", experiment_name)
    else:
        stats_dir = os.path.join(logs_dir, "stats")
        checkpoints_dir = os.path.join(logs_dir, "checkpoints")
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    tensorboard_writer = SummaryWriter(log_dir=stats_dir)

    early_stopper = EarlyStopper(patience=early_stopper_patience, min_delta=0)
    stop_training = False
    
    ema_model = None
    if use_ema:
        ema = EMA(beta=ema_decay)
        ema_model = copy.deepcopy(model)

    save_model_to_disk(model, 0, 0, checkpoints_dir)
    if ema_model is not None:
        save_model_to_disk(ema_model, 0, 0, checkpoints_dir)

    try:
        with tqdm(
            total=len(train_dataloader) * epochs, mininterval=1 if debug else 60
        ) as pbar:
            train_losses_l = []
            validation_losses_l = []
            for epoch in range(epochs):
                model.train()
                for _, train_batch_dict in enumerate(train_dataloader):
                    with TimerCUDA() as t_training_loss:
                        train_loss, train_losses_log = train_step(
                            model,
                            train_batch_dict,
                            loss_fn,
                            train_subset,
                            optimizer,
                            scaler,
                            use_amp,
                            clip_grad,
                            clip_grad_max_norm,
                            ema_model,
                            ema,
                            step,
                            ema_warmup,
                            ema_update_interval,
                            tensor_args,
                        )

                    if step % log_interval == 0:
                        print(f"\n-----------------------------------------")
                        print(f"step: {step}")
                        print(f"t_training_loss: {t_training_loss.elapsed:.4f} sec")
                        print(f"Total training loss {train_loss:.4f}")
                        print(f"Training losses {train_losses_log}")

                        train_losses_l.append((step, train_losses_log))

                        with TimerCUDA() as t_training_summary:
                            log(
                                step,
                                ema_model if ema_model is not None else model,
                                train_subset,
                                train_losses=train_losses_log,
                                prefix="TRAINING ",
                                debug=debug,
                                tensorboard_writer=tensorboard_writer,
                            )
                        print(f"t_training_summary: {t_training_summary.elapsed:.4f} sec")
                        
                        validation_losses_log = {}
                        if val_dataloader is not None:
                            with TimerCUDA() as t_validation_loss:
                                total_val_loss, validation_losses_log = val_step(
                                    model,
                                    val_dataloader,
                                    loss_fn,
                                    val_subset,
                                    step,
                                    steps_per_validation,
                                    tensor_args,
                                )

                            print(f"t_validation_loss: {t_validation_loss.elapsed:.4f} sec")
                            print(f"Validation losses {validation_losses_log}")

                            validation_losses_l.append(
                                (step, validation_losses_log)
                            )

                            with TimerCUDA() as t_validation_summary:
                                log(
                                    step,
                                    ema_model if ema_model is not None else model,
                                    val_subset,
                                    val_losses=validation_losses_log,
                                    prefix="VALIDATION ",
                                    debug=debug,
                                    tensorboard_writer=tensorboard_writer,
                                )
                            print(
                                f"t_valididation_summary: {t_validation_summary.elapsed:.4f} sec"
                            )

                    if val_dataloader is not None and early_stopper(total_val_loss):
                        print(f"Early stopped training at {step} steps.")
                        stop_training = True
                    stop_training |= step == max_steps

                    pbar.update(1)
                    step += 1

                    if (checkpoint_interval is not None) and (step % checkpoint_interval == 0):
                        save_model_to_disk(model, epoch, step, checkpoints_dir)
                        if ema_model is not None:
                            save_model_to_disk(ema_model, epoch, step, checkpoints_dir)
                        save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)

                    if stop_training:
                        break

                if stop_training:
                    break

            end_training(model, ema_model, ema, epoch, step, ema_warmup,
                        train_losses_l, validation_losses_l, checkpoints_dir)
            
            print(f"\n------- TRAINING FINISHED -------")

    except KeyboardInterrupt:
        print(f"\n\n------- TRAINING INTERRUPTED BY USER -------")
        print(f"Saving checkpoint at step {step}...")
        
        end_training(model, ema_model, ema, epoch, step, ema_warmup,
                    train_losses_l, validation_losses_l, checkpoints_dir, tensorboard_writer)
        
        print(f"------- TRAINING STOPPED -------\n")
        raise
