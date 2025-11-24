import torch

from mpd.losses.gaussian_diffusion_loss import gaussian_diffusion_loss_fn

# Default tensor arguments for device and dtype consistency
DEFAULT_TENSOR_ARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


# ==================== TRAINING CONFIGURATION ====================

LOSS_FNS = {
    "gaussian_diffusion_loss_fn": gaussian_diffusion_loss_fn,
}

DEFAULT_TRAIN_ARGS = {
    # Dataset
    "dataset_dir": "EnvSimple2D-RobotPointMass",
    "include_velocity": True,
    # Diffusion Model
    "diffusion_model_class": "GaussianDiffusionModel",
    "variance_schedule": "exponential",
    "n_diffusion_steps": 25,
    "predict_epsilon": True,
    # Unet
    "unet_input_dim": 32,
    "unet_dim_mults_option": 1,
    # Loss
    "loss_fn": "gaussian_diffusion_loss_fn",
    # Training parameters
    "num_train_steps": 500000,
    "lr": 1e-4,
    "batch_size": 32,
    "clip_grad": False,
    "clip_grad_max_norm": 1.0,
    "use_amp": False,
    "use_ema": True,
    "ema_decay": 0.995,
    "ema_warmup": 100000,
    "ema_update_interval": 10,
    "early_stopper_patience": -1,
    # Validation parameters
    "steps_per_validation": 10,
    # Summary parameters
    "log_interval": 1000,
    "checkpoint_interval": 50000,
    # Other
    "device": "cuda",
    "debug": False,
    "seed": 0,
    "logs_dir": "logs",
    "experiment_name": None,
}

