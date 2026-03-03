import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drmp.utils import load_config_from_yaml, fix_random_seed
from drmp.visualizer import Visualizer
from drmp.universe.environments import get_envs
from drmp.universe.robot import get_robots

def main():
    parser = argparse.ArgumentParser(description="Visualize trajectories from a .pt file.")
    parser.add_argument("trajectory_file", type=str, help="Path to the trajectories .pt file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image file path. Defaults to trajectory_file_name.png")
    parser.add_argument("--indices", "-i", type=int, nargs="+", default=None, help="Indices of trajectories to visualize")
    
    args = parser.parse_args()

    traj_path = Path(args.trajectory_file)
    if not traj_path.exists():
        print(f"Error: File {traj_path} does not exist.")
        return

    if args.output is None:
        args.output = str(traj_path.with_suffix('.png'))

    # Infer dataset directory (parent of the file)
    dataset_dir = traj_path.parent
    init_config_path = dataset_dir / "init_config.yaml"
    info_config_path = dataset_dir / "info_config.yaml"

    if not init_config_path.exists():
         print(f"Error: {init_config_path} not found. Cannot load environment configuration.")
         return

    init_config = load_config_from_yaml(str(init_config_path))
    
    seed = 0
    if info_config_path.exists():
        info_config = load_config_from_yaml(str(info_config_path))
        if "seed" in info_config:
            seed = info_config["seed"]
            print(f"Using seed from info_config: {seed}")
        else:
            print("Warning: Seed not found in info_config.yaml. Using default seed 0.")
    else:
        print("Warning: info_config.yaml not found. Using default seed 0.")

    # Fix seed to ensure environment generation matches dataset generation
    fix_random_seed(seed)
    
    # Setup environment and robot
    tensor_args = {"device": "cpu", "dtype": torch.float32}
    
    ENVS = get_envs()
    ROBOTS = get_robots()
    
    env_name = init_config["env_name"]
    robot_name = init_config["robot_name"]
    
    print(f"Loading environment: {env_name}, Robot: {robot_name}")

    if env_name not in ENVS:
        print(f"Error: Environment {env_name} not found in available environments.")
        return

    # Re-create environment with the same seed
    env = ENVS[env_name](tensor_args=tensor_args)
    
    robot_params = {
        "margin": init_config["robot_margin"],
        "dt": init_config["duration"] / (init_config["n_support_points"] - 1),
        "spline_degree": init_config["spline_degree"],
        "tensor_args": tensor_args,
    }
    robot_params.update(init_config["additional_robot_args"])
    
    if robot_name not in ROBOTS:
        print(f"Error: Robot {robot_name} not found in available robots.")
        return

    robot = ROBOTS[robot_name](**robot_params)

    # Load trajectories
    print(f"Loading trajectories from {traj_path}")
    try:
        trajectories = torch.load(traj_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading trajectory file: {e}")
        return
    
    # Check dimensions
    # Trajectories should be (N, support_points, dim)
    if not isinstance(trajectories, torch.Tensor):
         print(f"Error: Loaded object is not a tensor, but {type(trajectories)}")
         return

    if len(trajectories.shape) != 3:
        print(f"Error: Expected trajectories shape (N, support_points, dim), got {trajectories.shape}")
        return

    # Extract start and goal from the first trajectory
    # Assuming all trajectories correspond to the same task
    start_pos = trajectories[0, 0, :]
    goal_pos = trajectories[0, -1, :]
    
    visualizer = Visualizer(env=env, robot=robot, use_extra_objects=False)
    
    print(f"Visualizing {len(trajectories)} trajectories...")
    
    try:
        visualizer.render_scene(
            trajectories=trajectories,
            start_pos=start_pos,
            goal_pos=goal_pos,
            save_path=args.output,
            draw_indices=args.indices
        )
        print(f"Visualization saved to {args.output}")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
