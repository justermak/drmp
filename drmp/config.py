import os

import torch

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

N_DIM = 2

DEFAULT_TENSOR_ARGS = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

DEFAULT_TRAIN_ARGS = {
    "checkpoints_dir": os.path.join(dir_path, "models"),
    "checkpoint_name": None,
    # Dataset
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvDense2D_2000_50",
    "normalizer_name": "TrivialNormalizer",
    "n_control_points": 24,
    "spline_degree": 3,
    "apply_augmentations": True,
    "filter_collision": True,
    "filter_longest_portion": None,
    "filter_sharpest_portion": None,
    # Diffusion Model
    "diffusion_model_name": "DiffusionSplinesShortcut",  # "DiffusionDense", "DiffusionSplines", "DiffusionSplinesShortcut"
    "n_diffusion_steps": 32,
    "predict_epsilon": True,
    "ddim": False,
    "shortcut_steps": 8,
    # Unet
    "state_dim": N_DIM,
    "horizon": 24,
    "hidden_dim": 64,
    "dim_mults": "(1, 2, 4)",
    "kernel_size": 5,
    "resnet_block_groups": 8,
    "positional_encoding": "sinusoidal",
    "positional_encoding_dim": 16,
    "attn_heads": 4,
    "attn_head_dim": 32,
    "context_dim": 2 * N_DIM,
    # Training
    "num_train_steps": 500000,
    "lr": 1e-4,
    "weight_decay": 0,
    "batch_size": 2048,
    "bootstrap_fraction": 0.25,
    "clip_grad_max_norm": 1.0,
    "use_amp": True,
    "use_ema": True,
    "ema_decay": 0.995,
    "ema_warmup": 100000,
    "ema_update_interval": 10,
    # Summary
    "log_interval": 2000,
    "checkpoint_interval": 50000,
    # Guide
    "t_start_guide": 0,
    "n_guide_steps": 2,
    "guide_lambda_obstacles": 5e-3,
    "guide_lambda_position": 3e-6,
    "guide_lambda_velocity": 3e-7,
    "guide_lambda_acceleration": 7e-7,
    "guide_max_grad_norm": 1.0,
    "guide_n_interpolate": 10,
    # Other
    "device": "cuda:6",
    "debug": False,
    "seed": 42,
}

DEFAULT_INFERENCE_ARGS = {
    "generations_dir": os.path.join(dir_path, "runs"),
    "experiment_name": None,
    "n_tasks": 200,
    "n_samples": 100,
    "splits": '("test",)',  # '("train", "val", "test")',
    # Algorithm selection
    "algorithm": "diffusion",  # Options: "diffusion", "mpd", "mpd-splines", "rrt-connect", "gpmp2-uninformative", "gpmp2-rrt-prior"
    # Dataset
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvDense2D_2000_50",
    "threshold_start_goal_pos": 1.5,
    "use_extra_objects": True,
    # Diffusion model
    "checkpoints_dir": os.path.join(dir_path, "models", "checkpoints"),
    "checkpoint_name": "DiffusionSplines__EnvDense2D_2000_50__bs_1024__lr_0.0001__steps_300000__diffusion-steps_25__20260126_230039",
    "ddim": False,
    "shortcut_steps": 4,
    "guide_dense": False,
    # MPD guide
    # "t_start_guide": 6,
    # "n_guide_steps": 5,
    # "lambda_obstacles": 1e-2,
    # "lambda_gp": 2.5e-7,
    # "lambda_position": None, # 3e-6,
    # "lambda_velocity": None, # 3e-7,
    # "max_grad_norm": 1.0,
    # "n_interpolate": 5,
    # Diffusion prior guide
    "t_start_guide": 0,
    "n_guide_steps": 2,
    "lambda_obstacles": 5e-3,
    "lambda_position": 3e-6,
    "lambda_velocity": 3e-7,
    "lambda_acceleration": 7e-7,
    "max_grad_norm": 1.0,
    "n_interpolate": 10,
    # MPD
    "mpd_checkpoints_dir": os.path.join(
        dir_path,
        "data_trained_models",
        "EnvDense2D-RobotPointMass",
        "mpd",
        "checkpoints",
    ),
    "mpd_checkpoint_name": "ema_model_current.pth",
    "mpd_ddim": False,
    "mpd_start_guide_steps_fraction": 0.25,
    "mpd_n_guide_steps": 5,
    "mpd_sigma_collision": 1e1,
    "mpd_sigma_gp": 2e3,
    "mpd_max_grad_norm": 1.0,
    "mpd_n_interpolate": 5,
    # MPD-Splines
    "mpd_splines_checkpoints_dir": os.path.join(
        dir_path,
        "data_trained_models",
        "EnvDense2D-RobotPointMass",
        "mpd-splines",
        "checkpoints",
    ),
    "mpd_splines_checkpoint_name": "ema_model_current.pth",
    "mpd_splines_n_control_points": 24,
    "mpd_splines_spline_degree": 3,
    "mpd_splines_n_guide_steps": 4,
    "mpd_splines_start_guide_steps_fraction": 0.3,
    "mpd_splines_ddim_sampling_timesteps": 15,
    "mpd_splines_guide_lr": 1.0,
    "mpd_splines_scale_grad_prior": 0.25,
    "mpd_splines_sigma_collision": 1e1,
    "mpd_splines_sigma_gp": 2e3,
    "mpd_splines_max_grad_norm": 1.0,
    "mpd_splines_n_interpolate": 5,
    "mpd_splines_ddim": True,
    # Classical algorithm
    "classical_n_dof": N_DIM,
    "classical_sample_steps": 10000,
    "classical_opt_steps": 300,
    "classical_smoothen": True,
    # RRT-Connect parameters
    "rrt_connect_max_step_size": 0.005,
    "rrt_connect_max_radius": 0.3,
    "rrt_connect_n_samples": 160000,
    # GPMP2 parameters
    "gpmp2_n_interpolate": 5,
    "gpmp2_num_samples": 64,
    "gpmp2_sigma_start": 3e-2,
    "gpmp2_sigma_goal_prior": 3e-2,
    "gpmp2_sigma_gp": 1,
    "gpmp2_sigma_collision": 3e-3,
    "gpmp2_step_size": 1e-1,
    "gpmp2_delta": 1e-5,
    "gpmp2_method": "cholesky",
    # Other
    "device": "cuda",
    "debug": True,
    "seed": 42,
}

DEFAULT_DATA_GENERATION_ARGS = {
    # Dataset initialization
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvDense2D_2000_50",
    "env_name": "EnvDense2D",
    "robot_name": "Sphere2D",
    "robot_margin": 0.01,
    "generating_robot_margin": 0.02,
    "n_support_points": 64,
    "duration": 5.0,
    # Task generation
    "n_tasks": 2000,
    "n_trajectories": 50,
    "threshold_start_goal_pos": 1.5,
    # Planning parameters
    "sample_steps": 10000,
    "opt_steps": 300,
    # RRT-Connect parameters
    "rrt_connect_max_step_size": 0.005,
    "rrt_connect_max_radius": 0.3,
    "rrt_connect_n_samples": 160000,
    # GPMP2 parameters
    "gpmp2_n_interpolate": 5,
    "gpmp2_num_samples": 64,
    "gpmp2_sigma_start": 3e-2,
    "gpmp2_sigma_goal_prior": 3e-2,
    "gpmp2_sigma_gp": 1,
    "gpmp2_sigma_collision": 3e-3,
    "gpmp2_step_size": 1e-1,
    "gpmp2_delta": 1e-5,
    "gpmp2_method": "cholesky",
    # Other
    "max_processes": 1,
    "val_portion": 0.1,
    "device": "cuda",
    "debug": True,
    "seed": 42,
}
