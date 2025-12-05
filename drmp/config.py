import torch

N_DIM = 2

DEFAULT_TENSOR_ARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

DEFAULT_TRAIN_ARGS = {
    "checkpoints_dir": "../models/",
    "checkpoint_name": "EnvDense2D_50_15_bs32_lr0.0001_steps100000",
    # Dataset
    "datasets_dir": "../datasets/",
    "dataset_name": "EnvDense2D-50-15",
    # Diffusion Model
    "diffusion_model_name": "GaussianDiffusion",
    "variance_schedule": "exponential",
    "n_diffusion_steps": 25,
    "clip_denoised": True,
    "predict_epsilon": True,
    # Unet
    "hidden_dim": 32,
    "dim_mults": "(1, 2, 4, 8)",
    "kernel_size": 5,
    "resnet_block_groups": 8,
    "random_fourier_features": False,
    "learned_sin_dim": 16,
    "attn_heads": 4,
    "attn_head_dim": 32,
    "context_dim": 2 * N_DIM,
    # Training parameters
    "num_train_steps": 100000,
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
    "log_interval": 2000,
    "checkpoint_interval": 20000,
    # Other
    "device": "cuda",
    "debug": False,
    "seed": 0,
}

DEFAULT_INFERENCE_ARGS = {
    "generations_dir": "../runs/",
    "experiment_name": None,
    "n_tasks": 1,
    "n_samples": 50,
    "splits": '("train", "val", "test")',
    # Model
    "checkpoints_dir": "../models/checkpoints/",
    "checkpoint_name": "EnvDense2D_5_20_bs32_lr0.0001_steps1000",
    # Dataset
    "datasets_dir": "../datasets/",
    "dataset_name": "EnvDense2D-5-20",
    "threshold_start_goal_pos": 1.0,
    # Guide
    "use_extra_objects": True,
    # Sampling
    "n_diffusion_steps_without_noise": 5,
    "ddim": False,
    # Guidance
    "start_guide_steps_fraction": 0.25,
    "n_guide_steps": 5,
    "sigma_collision": 1e1,
    "sigma_gp": 1e3,
    "do_clip_grad": True,
    "max_grad_norm": 1.0,
    "do_interpolate": True,
    "n_interpolate": 5,
    # Other
    "device": "cuda",
    "debug": True,
    "seed": 0,
}

DEFAULT_DATA_GENERATION_ARGS = {
    # Dataset initialization
    "datasets_dir": "../datasets/",
    "dataset_name": "EnvDense2D-50-15",
    "env_name": "EnvDense2D",
    "normalizer_name": "LimitsNormalizer",
    "robot_margin": 0.01,
    "cutoff_margin": 0.02,
    "n_support_points": 64,
    "duration": 5.0,
    # Task generation
    "n_tasks": 50,
    "n_trajectories": 15,
    "threshold_start_goal_pos": 1.0,
    # Planning parameters
    "sample_steps": 10000,
    "opt_steps": 300,
    "use_parallel": True,
    "max_processes": -1,
    # Other
    "val_portion": 0.05,
    "device": "cuda",
    "debug": True,
    "seed": 0,
}
