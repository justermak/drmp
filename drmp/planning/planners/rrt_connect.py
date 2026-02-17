from typing import Any, Dict, List, Optional, Tuple

import torch

from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.torch_timer import TimerCUDA
from drmp.universe.environments import EnvBase
from drmp.universe.robot import RobotBase


class RRTConnect(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
        max_step_size: float,
        max_radius: float,
        n_samples: int,
        tensor_args: Dict[str, Any],
        n_trajectories: int = 1,
        use_extra_objects: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(
            env=env,
            robot=robot,
            use_extra_objects=use_extra_objects,
            tensor_args=tensor_args,
        )
        self.max_step_size = max_step_size
        self.max_radius = max_radius
        self.n_samples = n_samples
        self.n_trajectories = n_trajectories
        self.n_trajectories_with_luft = int(1.2 * n_trajectories)
        self.use_extra_objects = use_extra_objects
        self.eps = eps

        self.samples: torch.Tensor = None
        ok = self._initialize_samples()
        if not ok:
            raise RuntimeError("Failed to initialize RRTConnect samples")

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        self.start_pos = start_pos.to(**self.tensor_args)
        self.goal_pos = goal_pos.to(**self.tensor_args)

        self.samples_ptr = 0

        self.nodes_trees_1_pos = (
            self.start_pos.unsqueeze(0).repeat(self.n_trajectories_with_luft, 1).unsqueeze(1)
        )
        self.nodes_trees_2_pos = (
            self.goal_pos.unsqueeze(0).repeat(self.n_trajectories_with_luft, 1).unsqueeze(1)
        )
        self.nodes_trees_1_parents = [[None] for _ in range(self.n_trajectories_with_luft)]
        self.nodes_trees_2_parents = [[None] for _ in range(self.n_trajectories_with_luft)]

    def _initialize_samples(self) -> bool:
        samples, success = self.robot.random_collision_free_q(
            env=self.env,
            n_samples=self.n_samples,
            use_extra_objects=self.use_extra_objects,
        )
        if not success:
            print(
                "Could not find sufficient collision-free start/goal pairs for RRTConnect samples"
            )
        self.samples = samples
        return success

    def sample(self, n_samples) -> torch.Tensor:
        if self.samples_ptr + n_samples > len(self.samples):
            self._initialize_samples()
            self.samples_ptr = 0

        samples = self.samples[
            self.samples_ptr : self.samples_ptr + n_samples
        ]
        self.samples_ptr += n_samples
        return samples

    def get_nearest_nodes(
        self, nodes: torch.Tensor, targets: torch.Tensor, idxs: List[int]
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        distances = torch.linalg.norm(nodes[idxs] - targets.unsqueeze(1), dim=-1)
        min_idxs = torch.min(distances, dim=-1).indices
        return nodes[idxs, min_idxs], min_idxs

    def extend(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        max_step_size: float,
        max_radius: float,
    ) -> torch.Tensor:
        dist = torch.linalg.norm(q1 - q2, dim=-1)
        q2 = q1 + (q2 - q1) * torch.minimum(
            (max_radius / dist), torch.ones_like(dist)
        ).unsqueeze(-1)
        alpha = torch.linspace(0, 1, int(max_radius / max_step_size) + 2).to(
            **self.tensor_args
        )
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        extension = q1 + (q2 - q1) * alpha.unsqueeze(0).unsqueeze(-1)
        return extension

    def cut(self, sequences: torch.Tensor) -> torch.Tensor:
        in_collision = self.robot.get_collision_mask(
            env=self.env, qs=sequences, on_extra=self.use_extra_objects
        )
        in_collision = torch.cat(
            (in_collision, torch.ones_like(in_collision[:, :1], dtype=torch.bool)),
            dim=-1,
        )
        sequences = torch.cat((sequences, sequences[:, -1:, :]), dim=1)
        hack = torch.arange(sequences.shape[1], 0, -1).to(**self.tensor_args)
        max_idxs = torch.argmax(in_collision * hack, dim=-1) - 1
        res = sequences[torch.arange(sequences.shape[0]), max_idxs]
        return res

    def purge_duplicates_from_trajectories(
        self, paths: List[torch.Tensor]
    ) -> torch.Tensor:
        selections = []
        for path in paths:
            if path is None:
                print("WTF: RRT-Connect failed to find a path")
                selections.append(None)
                continue
            if path.shape[0] <= 2:
                selections.append(path)
                continue

            diff = torch.norm(torch.diff(path, dim=-2), dim=-1)
            idxs = torch.argwhere(diff > self.eps).squeeze(-1)
            selection = path[idxs]

            if not torch.allclose(selection[0], path[0], atol=self.eps):
                selection = torch.cat((path[:1], selection), dim=0)
            if not torch.allclose(selection[-1], path[-1], atol=self.eps):
                selection = torch.cat((selection, path[-1:]), dim=0)
            selections.append(selection)
        return selections

    def trace(
        self, nodes: torch.Tensor, parents: List[Optional[int]]
    ) -> List[torch.Tensor]:
        path = []
        idx = len(nodes) - 1
        while idx is not None:
            path.append(nodes[idx].unsqueeze(0))
            idx = parents[idx]
        return path

    def optimize(
        self, n_sampling_steps: int, print_freq: int = 50, debug: bool = True
    ) -> Optional[torch.Tensor]:
        self.n_sampling_steps = n_sampling_steps
        step = 0
        n_success = 0
        paths = []
        idxs = set(range(self.n_trajectories_with_luft))
        idxs_list = list(idxs)
        with TimerCUDA() as t:
            while step < n_sampling_steps:
                if debug and step % print_freq == 0:
                    self.print_info(step, t.elapsed, n_success)

                step += 1

                self.nodes_trees_1_pos, self.nodes_trees_2_pos = (
                    self.nodes_trees_2_pos,
                    self.nodes_trees_1_pos,
                )
                self.nodes_trees_1_parents, self.nodes_trees_2_parents = (
                    self.nodes_trees_2_parents,
                    self.nodes_trees_1_parents,
                )
                targets = self.sample(len(idxs_list))
                nearest_pos, nearest_idxs = self.get_nearest_nodes(
                    self.nodes_trees_1_pos, targets, idxs_list
                )
                extended = self.extend(
                    nearest_pos, targets, self.max_step_size, self.max_radius
                )
                cut = self.cut(extended)
                n1 = torch.zeros_like(self.nodes_trees_1_pos[..., 0, :])
                n1[idxs_list] = cut

                self.nodes_trees_1_pos = torch.cat(
                    [self.nodes_trees_1_pos, n1.unsqueeze(1)], dim=1
                )
                for i, j in enumerate(idxs_list):
                    self.nodes_trees_1_parents[j].append(nearest_idxs[i].item())

                nearest_pos, nearest_idxs = self.get_nearest_nodes(
                    self.nodes_trees_2_pos, cut, idxs_list
                )
                extended = self.extend(
                    nearest_pos, cut, self.max_step_size, self.max_radius
                )
                cut = self.cut(extended)
                n2 = torch.zeros_like(self.nodes_trees_2_pos[..., 0, :])
                n2[idxs_list] = cut

                self.nodes_trees_2_pos = torch.cat(
                    [self.nodes_trees_2_pos, n2.unsqueeze(1)], dim=1
                )
                for i, j in enumerate(idxs_list):
                    self.nodes_trees_2_parents[j].append(nearest_idxs[i].item())

                for i in idxs_list:
                    if torch.allclose(
                        self.nodes_trees_1_pos[i][-1],
                        self.nodes_trees_2_pos[i][-1],
                        atol=self.eps,
                    ):
                        path1 = self.trace(
                            self.nodes_trees_1_pos[i], self.nodes_trees_1_parents[i]
                        )
                        path2 = self.trace(
                            self.nodes_trees_2_pos[i], self.nodes_trees_2_parents[i]
                        )

                        tree_1_root = self.nodes_trees_1_pos[i][0]
                        if torch.allclose(tree_1_root, self.goal_pos, atol=self.eps):
                            path1, path2 = path2, path1

                        paths.append(torch.cat(path1[::-1] + path2[1:], dim=0))
                        n_success += 1
                        idxs.remove(i)
                        if n_success >= self.n_trajectories:
                            break
                        
                idxs_list = list(idxs)
                if n_success >= self.n_trajectories:
                    if debug:
                        self.print_info(step, t.elapsed, n_success)
                    break

        trajectories = self.purge_duplicates_from_trajectories(paths)
        return trajectories

    def print_info(self, step, elapsed_time, success) -> None:
        total_nodes = sum(
            len(self.nodes_trees_1_pos[i]) + len(self.nodes_trees_2_pos[i])
            for i in range(self.n_trajectories_with_luft)
        )
        pad = len(str(self.n_sampling_steps))
        print(
            f"| Step: {step:>{pad}}/{self.n_sampling_steps:>{pad}} "
            f"| Time: {elapsed_time:.3f} s"
            f"| Nodes: {total_nodes} "
            f"| Success: {success}/{self.n_trajectories}"
        )

    def visualize_trees(
        self,
        traj_idx: int = 0,
        save_path: str = "rrt_trees.png",
    ) -> None:
        """Visualize the RRT trees for a specific trajectory."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Render environment
        from drmp.visualizer import Visualizer

        vis = Visualizer(self.env, self.robot, use_extra_objects=self.use_extra_objects)
        vis._render_environment(ax, use_extra_objects=self.use_extra_objects)

        # Draw tree 1 (from start)
        nodes_1 = self.nodes_trees_1_pos[traj_idx].cpu().numpy()
        parents_1 = self.nodes_trees_1_parents[traj_idx]

        for i, parent_idx in enumerate(parents_1):
            if parent_idx is not None:
                child = nodes_1[i]
                parent = nodes_1[parent_idx]
                ax.plot(
                    [parent[0], child[0]],
                    [parent[1], child[1]],
                    "b-",
                    alpha=0.5,
                    linewidth=0.5,
                )

        ax.scatter(
            nodes_1[:, 0],
            nodes_1[:, 1],
            c="blue",
            s=10,
            alpha=0.6,
            label="Tree 1 (start)",
            zorder=10,
        )

        # Draw tree 2 (from goal)
        nodes_2 = self.nodes_trees_2_pos[traj_idx].cpu().numpy()
        parents_2 = self.nodes_trees_2_parents[traj_idx]

        for i, parent_idx in enumerate(parents_2):
            if parent_idx is not None:
                child = nodes_2[i]
                parent = nodes_2[parent_idx]
                ax.plot(
                    [parent[0], child[0]],
                    [parent[1], child[1]],
                    "r-",
                    alpha=0.5,
                    linewidth=0.5,
                )

        ax.scatter(
            nodes_2[:, 0],
            nodes_2[:, 1],
            c="red",
            s=10,
            alpha=0.6,
            label="Tree 2 (goal)",
            zorder=10,
        )

        # Draw start and goal
        start_np = self.start_pos.cpu().numpy()
        goal_np = self.goal_pos.cpu().numpy()
        ax.scatter(
            [start_np[0]],
            [start_np[1]],
            c="cyan",
            s=100,
            marker="*",
            label="Start",
            zorder=20,
            edgecolors="black",
        )
        ax.scatter(
            [goal_np[0]],
            [goal_np[1]],
            c="magenta",
            s=100,
            marker="*",
            label="Goal",
            zorder=20,
            edgecolors="black",
        )

        ax.legend(loc="upper right")
        ax.set_title(f"RRT-Connect Trees (Trajectory {traj_idx})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Tree visualization saved to {save_path}")
