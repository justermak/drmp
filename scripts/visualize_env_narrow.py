import os
import sys

import matplotlib.pyplot as plt
import torch

# Add the project root to the python path so we can import drmp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drmp.universe.environments import EnvDenseNarrowPassage2D
from drmp.universe.robot import RobotSphere2D
from drmp.visualizer import Visualizer


def main():
    # Setup tensor args
    device = "cpu"
    tensor_args = {"device": device, "dtype": torch.float32}

    print(f"Creating environment on {device}...")
    env = EnvDenseNarrowPassage2D(tensor_args=tensor_args)

    print("Creating robot...")
    robot = RobotSphere2D(
        margin=0.05, dt=0.01, spline_degree=3, tensor_args=tensor_args
    )

    print("Creating visualizer...")
    vis = Visualizer(env, robot)

    print("Rendering environment...")
    fig, ax = plt.subplots(figsize=(10, 10))
    vis._render_environment(ax)

    # Set title
    ax.set_title("EnvDenseNarrowPassage2D")

    output_file = "narrow_passage_env.png"
    print(f"Saving to {output_file}...")
    fig.savefig(output_file)
    print("Done.")


if __name__ == "__main__":
    main()
