from typing import List, Optional

import matplotlib.collections as mcoll
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import BoxStyle, FancyBboxPatch

from drmp.universe.environments import EnvBase
from drmp.universe.primitives import MultiBoxField, MultiSphereField
from drmp.universe.robot import RobotBase


class Visualizer:
    COLORS = {
        "collision": "black",
        "free": "orange",
        "robot_collision": "black",
        "robot_free": "darkorange",
        "robot_collision_moving": "red",
        "robot_free_moving": "blue",
        "traj_best": "green",
        "start": "cyan",
        "goal": "magenta",
        "fixed_obstacle": "gray",
        "extra_obstacle": "red",
    }

    START_GOAL_RADIUS = 0.005

    def __init__(
        self, env: EnvBase, robot: RobotBase, use_extra_objects: bool = True
    ) -> None:
        self.env = env
        self.robot = robot
        self.use_extra_objects = use_extra_objects

    def _render_environment(self, ax: plt.Axes, use_extra_objects: bool = True) -> None:
        for field in self.env.obj_field_fixed.fields:
            self._render_primitive_field(ax, field, color=self.COLORS["fixed_obstacle"])

        if use_extra_objects:
            for field in self.env.obj_field_extra.fields:
                self._render_primitive_field(
                    ax, field, color=self.COLORS["extra_obstacle"]
                )
        limits = self.env.limits_np
        ax.set_xlim(limits[0][0], limits[1][0])
        ax.set_ylim(limits[0][1], limits[1][1])

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def _render_primitive_field(
        self, ax: plt.Axes, field: MultiSphereField | MultiBoxField, color: str = "gray"
    ) -> None:
        if isinstance(field, MultiSphereField):
            centers_np = field.centers.cpu().numpy()
            radii_np = field.radii.cpu().numpy()
            for center_np, radius_np in zip(centers_np, radii_np):
                circle = plt.Circle(
                    center_np,
                    radius_np,
                    color=color,
                    linewidth=0,
                    alpha=1,
                )
                ax.add_patch(circle)
        elif isinstance(field, MultiBoxField):
            centers_np = field.centers.cpu().numpy()
            half_sizes_np = field.half_sizes.cpu().numpy()
            radii_np = field.radii.cpu().numpy()
            for center_np, half_size_np, radius in zip(
                centers_np, half_sizes_np, radii_np
            ):
                corner = center_np - half_size_np

                box = FancyBboxPatch(
                    corner,
                    2 * half_size_np[0],
                    2 * half_size_np[1],
                    color=color,
                    boxstyle=BoxStyle.Round(pad=0.0, rounding_size=radius),
                )
                ax.add_patch(box)

    def _render_start_goal_pos(
        self, ax: plt.Axes, start_pos: torch.Tensor, goal_pos: torch.Tensor
    ) -> None:
        start_pos_np = start_pos.cpu().numpy()
        goal_pos_np = goal_pos.cpu().numpy()

        circle = plt.Circle(
            start_pos_np, self.START_GOAL_RADIUS, color=self.COLORS["start"], zorder=100
        )
        ax.add_patch(circle)

        circle = plt.Circle(
            goal_pos_np, self.START_GOAL_RADIUS, color=self.COLORS["goal"], zorder=100
        )
        ax.add_patch(circle)

    def _compute_robot_pos_colors(
        self,
        collision_mask: torch.Tensor,
        best_traj_idx: int = None,
        moving: bool = False,
    ) -> List[str]:
        colors = [
            (
                (self.COLORS["robot_collision_moving" if moving else "robot_collision"])
                if collision
                else (
                    self.COLORS["traj_best"]
                    if i == best_traj_idx
                    else (self.COLORS["robot_free_moving" if moving else "robot_free"])
                )
            )
            for i, row in enumerate(collision_mask)
            for collision in row
        ]
        return colors

    def _compute_trajectory_colors(
        self, trajectories_collision_mask: torch.Tensor, best_traj_idx: int = None
    ) -> List[str]:
        colors = [
            self.COLORS["traj_best"]
            if i == best_traj_idx
            else self.COLORS["collision"]
            if coll
            else self.COLORS["free"]
            for i, coll in enumerate(trajectories_collision_mask)
        ]
        return colors

    def _render_robot_pos(
        self,
        ax: plt.Axes,
        states: torch.Tensor,
        colors: List[str],
        moving: bool = False,
        zorder: int = 10,
    ) -> None:
        pos = self.robot.get_position(states).reshape(-1, 2)
        pos_np = pos.view(-1, 2).cpu().numpy()

        for p, color in zip(pos_np, colors):
            circle = plt.Circle(
                p,
                self.robot.margin * (1.0 if moving else 0.5),
                color=color,
                alpha=0.5,
                zorder=zorder,
            )
            ax.add_patch(circle)

    def _render_trajectories(
        self, ax: plt.Axes, trajectories: torch.Tensor, colors: List[str]
    ) -> None:
        trajectories_pos = self.robot.get_position(trajectories)
        trajectories_np = trajectories_pos.cpu().numpy()
        segments = [trajectories_np[i] for i in range(trajectories_np.shape[0])]
        line_collection = mcoll.LineCollection(segments, colors=colors)
        ax.add_collection(line_collection)

    def render_scene(
        self,
        trajectories: torch.Tensor,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        best_traj_idx: Optional[int] = None,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        points_collision_mask: Optional[torch.Tensor] = None,
        save_path: Optional[str] = "trajectories_figure.png",
    ):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        B, N, S = trajectories.shape
        if points_collision_mask is None:
            _, _, points_collision_mask = (
                self.robot.get_trajectories_collision_and_free(
                    env=self.env,
                    trajectories=trajectories,
                    on_extra=self.use_extra_objects,
                )
            )
        trajectories_collision_mask = points_collision_mask.any(dim=-1)
        points_collision_mask = points_collision_mask[
            :, ::6
        ]  # from interpolated to original trajectories
        assert points_collision_mask.shape == (B, N)

        traj_colors = self._compute_trajectory_colors(
            trajectories_collision_mask, best_traj_idx
        )
        robot_pos_colors = self._compute_robot_pos_colors(
            points_collision_mask, best_traj_idx
        )

        self._render_environment(ax, use_extra_objects=self.use_extra_objects)
        self._render_trajectories(ax, trajectories, traj_colors)
        self._render_robot_pos(ax, trajectories, robot_pos_colors)

        if start_pos is not None and goal_pos is not None:
            self._render_start_goal_pos(ax, start_pos, goal_pos)

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig, ax

    def animate_robot_motion(
        self,
        trajectories: torch.Tensor,
        best_traj_idx: Optional[int] = None,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        n_frames: int = 60,
        anim_time: int = 5,
        save_path: str = "robot_motion_animation.mp4",
    ):
        B, N, S = trajectories.shape
        frame_indices = np.round(np.linspace(0, N - 1, n_frames)).astype(int)
        _, _, points_collision_mask = self.robot.get_trajectories_collision_and_free(
            env=self.env, trajectories=trajectories, on_extra=self.use_extra_objects
        )

        fig, ax = plt.subplots()

        def update_frame(frame_idx):
            idx = frame_indices[frame_idx]
            ax.clear()
            ax.set_title(f"Step: {idx}/{N - 1}")

            self.render_scene(
                fig=fig,
                ax=ax,
                trajectories=trajectories,
                best_traj_idx=best_traj_idx,
                start_pos=start_pos,
                goal_pos=goal_pos,
                points_collision_mask=points_collision_mask,
                save_path=None,
            )

            moving_colors = self._compute_robot_pos_colors(
                points_collision_mask[:, [idx]], moving=True
            )
            self._render_robot_pos(
                ax, trajectories[:, [idx], :], moving_colors, moving=True, zorder=20
            )

        self._save_animation(fig, update_frame, n_frames, anim_time, save_path)

    def animate_optimization_iterations(
        self,
        trajectories: torch.Tensor,
        best_traj_idx: Optional[int] = None,
        start_pos: Optional[torch.Tensor] = None,
        goal_pos: Optional[torch.Tensor] = None,
        n_frames: int = 60,
        anim_time: int = 5,
        save_path: str = "trajectories_optimization_animation.mp4",
    ):
        I, B, N, S = trajectories.shape
        frame_indices = np.round(np.linspace(0, I - 1, n_frames)).astype(int)

        _, _, points_collision_mask = self.robot.get_trajectories_collision_and_free(
            env=self.env,
            trajectories=trajectories.reshape(-1, N, S),
            on_extra=self.use_extra_objects,
        )
        points_collision_mask = points_collision_mask.reshape(I, B, -1)

        fig, ax = plt.subplots()

        def update_frame(frame_idx):
            idx = frame_indices[frame_idx]
            ax.clear()
            ax.set_title(f"Iteration: {frame_indices[frame_idx]}/{S - 1}")

            self.render_scene(
                fig=fig,
                ax=ax,
                trajectories=trajectories[idx],
                best_traj_idx=best_traj_idx if frame_idx == n_frames - 1 else None,
                start_pos=start_pos,
                goal_pos=goal_pos,
                points_collision_mask=points_collision_mask[idx],
                save_path=None,
            )

            self._render_start_goal_pos(ax, start_pos, goal_pos)

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
        writer = FFMpegWriter(
            fps=fps, codec="libx264", extra_args=["-preset", "ultrafast", "-crf", "23"]
        )
        animation.save(save_path, writer=writer, dpi=90)

        print(f"Animation saved to {save_path}")
