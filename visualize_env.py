
import argparse
import matplotlib.pyplot as plt
import torch
import sys
import os
import math

# Add current directory to path so we can import drmp
sys.path.append(os.getcwd())

from drmp.universe.environments import get_envs
from drmp.universe.robot import get_robots
from drmp.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Render an environment using Visualizer")
    parser.add_argument("env_name", type=str, help="Name of the environment to render")
    parser.add_argument("--robot_name", type=str, default="L2D", help="Name of the robot (default: Sphere2D)")
    parser.add_argument("--robot_margin", type=float, default=0.05, help="Robot margin (radius)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--pos_x", type=float, default=None, help="Robot X position for visualization")
    parser.add_argument("--pos_y", type=float, default=None, help="Robot Y position for visualization")
    parser.add_argument("--pos_theta", type=float, default=0.0, help="Robot Theta (radians) for visualization (for L2D)")

    args = parser.parse_args()

    # Default L2D params from info_config.yaml
    l2d_params = {
        "width": 0.3,
        "height": 0.4,
        "n_spheres": 15
    }

    if "EnvSparse" in args.env_name:
        print(f"Detected EnvSparse ({args.env_name}), using L2D robot with default parameters.")
        args.robot_name = "L2D"
        args.robot_margin = 0.05 

    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}

    envs = get_envs()
    if args.env_name not in envs:
        print(f"Error: Environment '{args.env_name}' not found. Available environments: {list(envs.keys())}")
        return

    robots = get_robots()
    if args.robot_name not in robots:
        print(f"Error: Robot '{args.robot_name}' not found. Available robots: {list(robots.keys())}")
        return

    print(f"Initializing Environment: {args.env_name}")
    env_class = envs[args.env_name]
    env = env_class(tensor_args=tensor_args)

    print(f"Initializing Robot: {args.robot_name}")
    robot_class = robots[args.robot_name]
    
    # Basic robot args
    robot_args = {
        "margin": args.robot_margin,
        "dt": 0.1, # Dummy value, not needed for env rendering
        "spline_degree": 3, # Dummy value
        "tensor_args": tensor_args
    }
    
    # Add dummy additional args for L2D if needed, though default init might handle it or crash
    if args.robot_name == "L2D":
         robot_args.update({"width": 0.3, "height": 0.4, "n_spheres": 15})

    try:
        robot = robot_class(**robot_args)
    except TypeError as e:
        print(f"Error initializing robot: {e}")
        # If it fails, maybe it needs specific args not provided. 
        # But for Sphere2D it should work with base args.
        return

    viz = Visualizer(env=env, robot=robot)

    fig, ax = plt.subplots(figsize=(8, 8))
    viz._render_environment(ax)

    if args.pos_x is not None and args.pos_y is not None:
        pos_x = args.pos_x
        pos_y = args.pos_y
        theta = args.pos_theta

        state = None
        if args.robot_name == "L2D":
            # Construct L2D state
            w = l2d_params["width"]
            h = l2d_params["height"]
            
            rx = pos_x + w * math.cos(theta)
            ry = pos_y + w * math.sin(theta)
            
            tx = pos_x - h * math.sin(theta)
            ty = pos_y + h * math.cos(theta)
            
            state_list = [rx, ry, pos_x, pos_y, tx, ty]
            state = torch.tensor([[state_list]], **tensor_args) # (1, 1, 6)
            
        elif args.robot_name == "Sphere2D":
            state = torch.tensor([[[pos_x, pos_y]]], **tensor_args) # (1, 1, 2)
            
        if state is not None:
            # Colors: List[List[str]]
            # 1 trajectory, 1 time step
            colors = [[viz.COLORS["robot_free"]]]
            
            viz._render_robot_pos(
                ax,
                trajectories=state,
                colors=colors,
                type="base", # use base style
                zorder=viz.ZORDERS["robot"]
            )
            
            # Maybe add a label or marker for the position
            ax.plot(pos_x, pos_y, 'ko', markersize=2, zorder=viz.ZORDERS["robot"] + 1)
    
    ax.set_title(f"Environment: {args.env_name}")
    plt.tight_layout()
    plt.savefig(f"{args.env_name}_render.png")

if __name__ == "__main__":
    main()
