import os
from datetime import datetime
from copy import copy

import configargparse
import torch
from torch.utils.data import Subset
from einops._torch_specific import allow_ops_in_compiled_graph

from drmp.config import DEFAULT_INFERENCE_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.inference import run_inference
from drmp.models.models import get_models
from drmp.utils.seed import fix_random_seed
from drmp.utils.torch_utils import freeze_torch_model_params
from drmp.utils.yaml import load_config_from_yaml

allow_ops_in_compiled_graph()

MODELS = get_models()


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}
    
    print(f"-------- INFERENCE STARTED --------")
    print(f"model: {args.checkpoint_name}")
    print(f"dataset: {args.dataset_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_samples: {args.n_samples}")

    checkpoint_dir = os.path.join(args.checkpoints_dir, args.checkpoint_name)
    model_config_path = os.path.join(checkpoint_dir, "config.yaml")
    model_config = load_config_from_yaml(model_config_path)

    dataset_dir = os.path.join(args.datasets_dir, args.dataset_name)
    dataset_config_path = os.path.join(dataset_dir, "config.yaml")
    dataset_config = load_config_from_yaml(dataset_config_path)
    dataset_init_config = {
        "datasets_dir": args.datasets_dir,
        "dataset_name": args.dataset_name,
        "normalizer_name": dataset_config["normalizer_name"],
        "robot_margin": dataset_config["robot_margin"],
        "env_name": dataset_config["env_name"],
        "tensor_args": tensor_args,
    }
    dataset = TrajectoryDataset(**dataset_init_config)
    dataset.load_data()
    train_subset, _, val_subset, _ = dataset.load_train_val_split()

    splits = eval(args.splits)
    
    test_subset = None
    
    if "test" in splits:
        task = dataset.task

        start_pos, goal_pos, success = task.env.random_coll_free_start_goal(
            n_samples=args.n_tasks,
            threshold_start_goal_pos=args.threshold_start_goal_pos,
        )
        if not success:
            print("Could not find sufficient collision-free start/goal pairs for test tasks, try reducing the threshold, robot margin or object density")
            return
        test_dataset = copy(dataset)
        test_dataset.n_trajs = args.n_tasks
        test_dataset.trajs_normalized = torch.empty((args.n_tasks,), **tensor_args)
        test_dataset.start_states = torch.cat([start_pos, torch.zeros_like(start_pos)], dim=-1)
        test_dataset.goal_states = torch.cat([goal_pos, torch.zeros_like(goal_pos)], dim=-1)
        test_dataset.start_states_normalized = test_dataset.normalizer.normalize(test_dataset.start_states)
        test_dataset.goal_states_normalized = test_dataset.normalizer.normalize(test_dataset.goal_states)
        test_subset = Subset(test_dataset, list(range(args.n_tasks)))
        
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.checkpoint_name}_{args.dataset_name}_{timestamp}"
    else:
        experiment_name = args.experiment_name


    model = MODELS[model_config["model_name"]](
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=model_config["unet_input_dim"],
        unet_dim_mults=eval(model_config["unet_dim_mults"]),
        variance_schedule=model_config["variance_schedule"],
        n_diffusion_steps=model_config["n_diffusion_steps"],
        predict_epsilon=model_config["predict_epsilon"],
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(
                checkpoint_dir,
                "ema_model_current_state_dict.pth"
                if model_config["use_ema"]
                else "model_current_state_dict.pth",
            ),
            map_location=tensor_args["device"],
        )
    )
    model.eval()
    freeze_torch_model_params(model)
    model = torch.compile(model)

    run_inference(
        model=model,
        dataset=dataset,
        train_subset=train_subset if "train" in splits else None,
        val_subset=val_subset if "val" in splits else None,
        test_subset=test_subset,
        generations_dir=args.generations_dir,
        experiment_name=experiment_name,
        trajectory_duration=args.trajectory_duration,
        use_extra_objects=args.use_extra_objects,
        weight_grad_cost_collision=args.weight_grad_cost_collision,
        weight_grad_cost_smoothness=args.weight_grad_cost_smoothness,
        num_interpolated_points_for_collision=args.num_interpolated_points_for_collision,
        start_guide_steps_fraction=args.start_guide_steps_fraction,
        n_tasks=args.n_tasks,
        threshold_start_goal_pos=args.threshold_start_goal_pos,
        n_samples=args.n_samples,
        n_guide_steps=args.n_guide_steps,
        n_diffusion_steps_without_noise=args.n_diffusion_steps_without_noise,
        ddim=args.ddim,
        debug=args.debug,
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {}

    for key, value in DEFAULT_INFERENCE_ARGS.items():
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
