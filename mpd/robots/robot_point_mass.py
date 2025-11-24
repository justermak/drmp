import matplotlib.pyplot as plt
import numpy as np
import torch

from mpd.robots.robot_base import RobotBase
from mpd.utils.torch_utils import to_numpy, to_torch

import matplotlib.collections as mcoll


class RobotPointMass(RobotBase):
    def __init__(
        self,
        name="RobotPointMass",
        q_limits=torch.tensor([[-1, -1], [1, 1]]),
        tensor_args=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            q_limits=to_torch(q_limits, **tensor_args),
            link_margins_for_object_collision_checking=[0.01],
            link_idxs_for_object_collision_checking=[0],
            num_interpolated_points_for_object_collision_checking=1,
            tensor_args=tensor_args,
            **kwargs,
        )

    def fk_map_collision_impl(self, q, **kwargs):
        # There is no forward kinematics. Assume it's the identity.
        # Add tasks space dimension
        return q.unsqueeze(-2)

    def render(
        self, ax, q=None, color="blue", cmap="Blues", margin_multiplier=1.0, **kwargs
    ):
        if q is not None:
            margin = (
                self.link_margins_for_object_collision_checking[0] * margin_multiplier
            )
            q = to_numpy(q)
            if q.ndim == 1:
                assert self.q_dim == 2, "Only 2D rendering is supported"
                circle1 = plt.Circle(q, margin, color=color, zorder=10)
                ax.add_patch(circle1)
            elif q.ndim == 2:
                assert q.shape[-1] == 2, "Only 2D rendering is supported"
                circ = []
                for q_ in q:
                    circ.append(plt.Circle(q_, margin, color=color))
                coll = mcoll.PatchCollection(circ, zorder=10)
                ax.add_collection(coll)
            else:
                raise NotImplementedError

    def render_trajectories(
        self,
        ax,
        trajs=None,
        start_state=None,
        goal_state=None,
        colors=["blue"],
        linestyle="solid",
        **kwargs,
    ):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            trajs_np = to_numpy(trajs_pos)
            assert self.q_dim == 2, "Only 2D rendering is supported"
            segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(
                1, 2
            )
            line_segments = mcoll.LineCollection(
                segments, colors=colors, linestyle=linestyle
            )
            ax.add_collection(line_segments)
            points = np.reshape(trajs_np, (-1, 2))
            colors_scatter = []
            for segment, color in zip(segments, colors):
                colors_scatter.extend([color] * segment.shape[0])
            ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2)
        if start_state is not None:
            start_state_np = to_numpy(start_state)
            assert len(start_state_np) == 2, "Only 2D rendering is supported"
            ax.plot(start_state_np[0], start_state_np[1], "go", markersize=7)
        if goal_state is not None:
            goal_state_np = to_numpy(goal_state)
            assert len(goal_state_np) == 2, "Only 2D rendering is supported"
            ax.plot(
                goal_state_np[0],
                goal_state_np[1],
                marker="o",
                color="purple",
                markersize=7,
            )
