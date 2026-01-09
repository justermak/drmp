import os
import torch

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

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
    "dataset_name": "EnvDense2D_1000_100",
    "use_filtered_trajectories": True,
    # Diffusion Model
    "diffusion_model_name": "GaussianDiffusion",
    "n_diffusion_steps": 50,
    "predict_epsilon": True,
    # Unet
    "hidden_dim": 32,
    "dim_mults": "(1, 2, 4)",
    "kernel_size": 5,
    "resnet_block_groups": 8,
    "positional_encoding": "random_fourier",
    "positional_encoding_dim": 16,
    "attn_heads": 4,
    "attn_head_dim": 32,
    "context_dim": 2 * N_DIM,
    # Training parameters
    "num_train_steps": 20000,
    "lr": 2e-4,
    "batch_size": 1024,
    "clip_grad": True,
    "clip_grad_max_norm": 2.0,
    "use_amp": True,
    "use_ema": True,
    "ema_decay": 0.995,
    "ema_warmup": 100000,
    "ema_update_interval": 10,
    "early_stopper_patience": 10,
    # Summary parameters
    "log_interval": 2000,
    "checkpoint_interval": 20000,
    # Other
    "device": "cuda",
    "debug": False,
    "seed": 42,
}

DEFAULT_INFERENCE_ARGS = {
    "generations_dir": os.path.join(dir_path, "runs"),
    "experiment_name": None,
    "n_tasks": 100,
    "n_samples": 100,
    "splits": '("test",)', # '("train", "val", "test")',
    # Algorithm selection
    "algorithm": "gpmp2-rrt-prior",  # Options: "diffusion", "legacy-diffusion", "rrt-connect", "gpmp2-uninformative", "gpmp2-rrt-prior"
    # Dataset
    "datasets_dir": os.path.join(dir_path, "datasets"),
    "dataset_name": "EnvDense2D_1000_100",
    "threshold_start_goal_pos": 1.5,
    "use_extra_objects": True,
    # Diffusion model
    "checkpoints_dir": os.path.join(dir_path, "models", "checkpoints"),
    "checkpoint_name": None,
    "ddim": False,
    "start_guide_steps_fraction": 0.25,
    "n_guide_steps": 5,
    "sigma_collision": 1e1,
    "sigma_gp": 2e3,
    "do_clip_grad": True,
    "max_grad_norm": 1.0,
    "n_interpolate": 5,
    # Legacy diffusion model
    "legacy_checkpoints_dir": os.path.join(dir_path, "data_trained_models", "EnvDense2D-RobotPointMass", "checkpoints"),
    "legacy_checkpoint_name": "ema_model_current.pth",
    "legacy_ddim": False,
    "legacy_start_guide_steps_fraction": 0.25,
    "legacy_n_guide_steps": 5,
    "legacy_sigma_collision": 1e1,
    "legacy_sigma_gp": 2e3,
    "legacy_do_clip_grad": True,
    "legacy_max_grad_norm": 1.0,
    "legacy_n_interpolate": 5,
    # Classical algorithm
    "classical_n_dof": N_DIM,
    "classical_sample_steps": 10000,
    "classical_opt_steps": 300,
    "classical_use_parallel": True,
    "classical_max_processes": 12,
    "classical_smoothen": True,
    # RRT-Connect parameters
    "rrt_connect_step_size": 0.005,
    "rrt_connect_n_radius": 0.3,
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
    "dataset_name": "EnvDense2D_1000_100",
    "env_name": "EnvDense2D",
    "normalizer_name": "LimitsNormalizer",
    "robot_margin": 0.01,
    "generating_robot_margin": 0.02,
    "n_support_points": 64,
    "duration": 5.0,
    # Task generation
    "n_tasks": 1000,
    "n_trajectories": 100,
    "threshold_start_goal_pos": 1.5,
    # Planning parameters
    "sample_steps": 10000,
    "opt_steps": 300,
    "use_parallel": True,
    "max_processes": 15,
    # RRT-Connect parameters
    "rrt_connect_step_size": 0.005,
    "rrt_connect_n_radius": 0.3,
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
    "val_portion": 0.05,
    "filter_longest_portion": 0.25,
    "filter_sharpest_portion": 0.25,
    "device": "cuda",
    "debug": False,
    "seed": 42,
}
