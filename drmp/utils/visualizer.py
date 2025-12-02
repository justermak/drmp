from typing import List, Optional
import matplotlib.collections as mcoll
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import BoxStyle, FancyBboxPatch
import torch

from drmp.utils.torch_utils import to_numpy
from drmp.world.environments import EnvBase
from drmp.world.primitives import MultiBoxField, MultiSphereField
from drmp.world.robot import Robot


class Visualizer:
    COLORS = {
        "collision": "black",
        "free": "orange",
        "robot_collision": "black",
        "robot_free": "darkorange",
        "traj_best": "green",
        "start": "cyan",
        "goal": "magenta",
        "fixed_obstacle": "gray",
        "extra_obstacle": "red",
    }
    
    START_GOAL_RADIUS = 0.05
    TRAJECTORY_POINT_SIZE = 4
    
    def __init__(self, env: EnvBase, robot: Robot) -> None:
        self.env: EnvBase = env
        self.robot: Robot = robot
    
    def _render_environment(self, ax: plt.Axes, plot_extra_objects: bool = True) -> None:
        for field in self.env.obj_field_fixed.fields:
            self._render_primitive_field(ax, field, color=self.COLORS["fixed_obstacle"])

        if plot_extra_objects:
            for field in self.env.obj_field_extra.fields:
                self._render_primitive_field(ax, field, color=self.COLORS["extra_obstacle"])
        limits = self.env.limits_np
        ax.set_xlim(limits[0][0], limits[1][0])
        ax.set_ylim(limits[0][1], limits[1][1])
        
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def _render_primitive_field(self, ax: plt.Axes, field: MultiSphereField|MultiBoxField, color: str ="gray") -> None:
        if isinstance(field, MultiSphereField):
            for center, radius in zip(field.centers, field.radii):
                center_np = to_numpy(center)
                radius_np = to_numpy(radius)
                
                circle = plt.Circle(
                    center_np,
                    radius_np,
                    color=color,
                    linewidth=0,
                    alpha=1,
                )
                ax.add_patch(circle)
        elif isinstance(field, MultiBoxField):
            for center, half_size, radius in zip(field.centers, field.half_sizes, field.radii):
                center_np = to_numpy(center)
                half_size_np = to_numpy(half_size)
                corner = center_np - half_size_np
                
                box = FancyBboxPatch(
                    corner,
                    2 * half_size_np[0],
                    2 * half_size_np[1],
                    color=color,
                    boxstyle=BoxStyle.Round(pad=0.0, rounding_size=to_numpy(radius).item()),
                )
                ax.add_patch(box)

    def _render_start_goal_states(self, ax: plt.Axes, start_state: torch.Tensor, goal_state: torch.Tensor) -> None:
        start_pos = to_numpy(start_state)
        circle = plt.Circle(
            start_pos,
            self.START_GOAL_RADIUS,
            color=self.COLORS["start"],
            zorder=100
        )
        ax.add_patch(circle)
        
        goal_pos = to_numpy(goal_state)
        circle = plt.Circle(
            goal_pos,
            self.START_GOAL_RADIUS,
            color=self.COLORS["goal"],
            zorder=100
        )
        ax.add_patch(circle)

    def _render_robot_states(self, ax: plt.Axes, states: torch.Tensor, colors: List[str] = None) -> None:          
        pos = self.robot.get_position(states).reshape(-1, 2)
        pos_np = to_numpy(pos)
        if colors is None:
            collision_mask = self.env.get_collision_mask(self.robot, pos)
            colors = [
                self.COLORS["robot_collision"] if coll else self.COLORS["robot_free"]
                for coll in collision_mask
            ]
        
        for p, color in zip(pos_np, colors):
            circle = plt.Circle(p, self.robot.margin, color=color, alpha=0.5, zorder=10)
            ax.add_patch(circle)

    def _render_trajectories(self, ax: plt.Axes, trajs: torch.Tensor, colors: List[str] = None) -> None:           
        _, _, points_collision_mask = self.env.get_trajs_collision_and_free(self.robot, trajs)
        trajs_collision_mask = points_collision_mask.any(dim=-1)
        if colors is None:
            colors = [
                self.COLORS["collision"] if coll else self.COLORS["free"]
                for coll in trajs_collision_mask
            ]   
        
        trajs_pos = self.robot.get_position(trajs)
        trajs_np = to_numpy(trajs_pos)
        segments = [trajs_np[i] for i in range(trajs_np.shape[0])]
        line_collection = mcoll.LineCollection(segments, colors=colors)
        ax.add_collection(line_collection)
        
    def render_scene(
        self,
        trajs: torch.Tensor,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        traj_best: Optional[torch.Tensor] = None,
        start_state: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None,
        save_path: Optional[str] = "trajectories_figure.png",
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        self._render_environment(ax)
        self._render_trajectories(ax, trajs)
        self._render_robot_states(ax, trajs)
        
        if traj_best is not None:
            self._render_trajectories(
                ax, 
                traj_best.unsqueeze(0), 
                colors=[self.COLORS["traj_best"]]
            )
            self._render_robot_states(
                ax, 
                traj_best.unsqueeze(0), 
                colors=[self.COLORS["traj_best"]]
            )

        self._render_start_goal_states(ax, start_state, goal_state)
        
        if save_path is not None:
            print("Saving figure...")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        return fig, ax

    def animate_trajectories(
        self,
        trajs: torch.Tensor,
        traj_best: Optional[torch.Tensor] = None,
        start_state: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None,
        n_frames: int = 60,
        anim_time: int = 5,
        save_path : str = "trajectories_animation.mp4",
    ):

        B, H, D = trajs.shape
        frame_indices = np.round(np.linspace(0, H - 1, n_frames)).astype(int)
        trajs_at_frames = trajs[:, frame_indices, :]

        fig, ax = plt.subplots()

        def update_frame(frame_idx):
            ax.clear()
            ax.set_title(f"Step: {frame_indices[frame_idx]}/{H - 1}")
            
            self.render_scene(
                fig=fig,
                ax=ax,
                trajs=trajs,
                traj_best=traj_best if frame_idx == n_frames - 1 else None,
                start_state=start_state,
                goal_state=goal_state,
                save_path=None,
            )

            current_states = trajs_at_frames[:, frame_idx, :]
            collision_mask = self.env.get_collision_mask(self.robot, current_states)
            colors = [
                self.COLORS["robot_collision"] if coll else self.COLORS["robot_free"]
                for coll in collision_mask
            ]
            self._render_robot_states(ax, current_states, colors=colors)

        self._save_animation(fig, update_frame, n_frames, anim_time, save_path)
    
    animate_robot_trajectories = animate_trajectories

    def animate_optimization_iterations(
        self,
        trajs: torch.Tensor,
        traj_best: Optional[torch.Tensor] = None,
        start_state: Optional[torch.Tensor] = None,
        goal_state: Optional[torch.Tensor] = None,
        n_frames: int = 60,
        anim_time: int = 5,
        save_path: str = "trajectories_optimization_animation.mp4",
    ):

        S, B, H, D = trajs.shape
        frame_indices = np.round(np.linspace(0, S - 1, n_frames)).astype(int)
        trajs_at_iters = trajs[frame_indices]

        fig, ax = plt.subplots()

        def update_frame(frame_idx):
            ax.clear()
            ax.set_title(f"Iteration: {frame_indices[frame_idx]}/{S - 1}")
            
            is_last_frame = frame_idx == n_frames - 1
            
            self.render_scene(
                fig=fig,
                ax=ax,
                trajs=trajs_at_iters[frame_idx],
                traj_best=traj_best if is_last_frame else None,
                start_state=start_state,
                goal_state=goal_state,
                save_path=None,
            )
            
            self._render_start_goal_states(ax, start_state, goal_state)

        self._save_animation(fig, update_frame, n_frames, anim_time, save_path)

    def _save_animation(self, fig, update_fn, n_frames, anim_time, save_path):
        print("Creating animation...")
        
        animation = FuncAnimation(
            fig,
            update_fn,
            frames=n_frames,
            interval=anim_time * 1000 / n_frames,
            repeat=False,
        )
        
        print("Saving animation...")
        fps = max(1, int(n_frames / anim_time))
        animation.save(save_path, fps=fps, dpi=90)
        
        print(f"Animation saved to {save_path}")
