import torch

N_DIMS = 2

DEFAULT_TENSOR_ARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

DEFAULT_TRAIN_ARGS = {
    "checkpoints_dir": "../models/",
    "checkpoint_name": "EnvDense2D_5_20_bs32_lr0.0001_steps1000",
    # Dataset
    "datasets_dir": "../datasets/",
    "dataset_name": "EnvDense2D-5-20",
    # Diffusion Model
    "diffusion_model_name": "GaussianDiffusion",
    "variance_schedule": "exponential",
    "n_diffusion_steps": 25,
    "predict_epsilon": True,
    # Unet
    "unet_input_dim": 32,
    "unet_dim_mults": "(1, 2, 4)",
    "time_emb_dim": 32,
    "self_attention": False,
    "conditioning_embed_dim": 4,
    "conditioning_type": None,
    "attention_num_heads": 2,
    "attention_dim_head": 32,
    # Training parameters
    "num_train_steps": 10000,
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
    "log_interval": 100,
    "checkpoint_interval": 200,
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
    "splits": "(\"train\", \"val\", \"test\")",
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
    "weight_grad_cost_collision": 1e-2,
    "weight_grad_cost_smoothness": 1e-7,
    "num_interpolated_points_for_collision": 5,
    # Trajectory
    "trajectory_duration": 5.0,
    # Other
    "device": "cuda",
    "debug": True,
    "seed": 0,
}

DEFAULT_DATA_GENERATION_ARGS = {
    # Dataset initialization
    "datasets_dir": "../datasets/",
    "dataset_name": "EnvDense2D-5-20",
    "env_name": "EnvDense2D",
    "normalizer_name": "LimitsNormalizer",
    "robot_margin": 0.01,
    "cutoff_margin": 0.01,
    # Task generation
    "n_tasks": 5,
    "n_trajectories": 5,
    "threshold_start_goal_pos": 1.0,
    # Planning parameters
    "sample_iters": 10000,
    "opt_iters": 500,
    # Trajectory parameters
    "n_support_points": 64,
    "duration": 5.0,
    # Other
    "val_portion": 0.05,
    "device": "cuda",
    "debug": True,
    "seed": 0,
}
