from typing import Any, Dict, List, Optional, Tuple

import torch

from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.utils.torch_timer import TimerCUDA
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot

class RRTConnect(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: Robot,
        max_step_size: float,
        max_radius: float,
        n_samples: int,
        n_trajectories: int,
        tensor_args: Dict[str, Any],
        use_extra_objects: bool = False,
        planner_id: int = None,
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
        self.n_trajectories = n_trajectories
        self.use_extra_objects = use_extra_objects
        self.planner_id = planner_id
        self.eps = eps

        self.n_samples = n_samples
        self.samples: torch.Tensor = None
        ok = self._initialize_samples()
        if not ok:
            raise RuntimeError("Failed to initialize RRTConnect samples")

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        self.start_pos = start_pos.to(**self.tensor_args)
        self.goal_pos = goal_pos.to(**self.tensor_args)

        self.samples_ptr = 0

        self.nodes_trees_1_pos = self.start_pos.unsqueeze(0).repeat(self.n_trajectories, 1).unsqueeze(1)
        self.nodes_trees_2_pos = self.goal_pos.unsqueeze(0).repeat(self.n_trajectories, 1).unsqueeze(1)
        self.nodes_trees_1_parents = [[None] for _ in range(self.n_trajectories)]
        self.nodes_trees_2_parents = [[None] for _ in range(self.n_trajectories)]

    def _initialize_samples(self) -> bool:
        samples, success = self.env.random_collision_free_q(
            robot=self.robot,
            n_samples=self.n_samples,
            use_extra_objects=self.use_extra_objects,
        )
        if not success:
            print(
                "Could not find sufficient collision-free start/goal pairs for RRTConnect samples"
            )
        self.samples = samples
        return success

    def sample(self) -> torch.Tensor:
        if self.samples_ptr + self.n_trajectories > len(self.samples):
            self._initialize_samples()
            self.samples_ptr = 0

        samples = self.samples[self.samples_ptr: self.samples_ptr + self.n_trajectories]
        self.samples_ptr += self.n_trajectories
        return samples

    def get_nearest_nodes(
        self, nodes: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        distances = torch.linalg.norm(nodes - targets.unsqueeze(1), dim=-1)
        min_idxs = torch.min(distances, dim=-1).indices
        return nodes[torch.arange(nodes.shape[0]), min_idxs], min_idxs

    def extend(
        self, q1: torch.Tensor, q2: torch.Tensor, max_step_size: float, max_radius: float
    ) -> torch.Tensor:
        dist = torch.linalg.norm(q1 - q2, dim=-1)
        q2 = (q1 + (q2 - q1) * torch.minimum((max_radius / dist), torch.ones_like(dist)).unsqueeze(-1))
        alpha = torch.linspace(0, 1, int(max_radius / max_step_size) + 2).to(**self.tensor_args)
        q1 = q1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        extension = q1 + (q2 - q1) * alpha.unsqueeze(0).unsqueeze(-1)
        return extension
    
    def cut(self, sequences: torch.Tensor) -> torch.Tensor:
        in_collision = self.env.get_collision_mask(
            self.robot, sequences, on_extra=self.use_extra_objects
        )
        in_collision = torch.cat((in_collision, torch.ones_like(in_collision[:, :1], dtype=torch.bool)), dim=-1)
        sequences = torch.cat((sequences, sequences[:, -1:, :]), dim=1)
        hack = torch.arange(sequences.shape[1], 0, -1).to(**self.tensor_args)
        idxs = torch.argmax(in_collision * hack, dim=-1) - 1
        res = sequences[torch.arange(sequences.shape[0]), idxs]
        return res

    def purge_duplicates_from_trajs(self, paths: List[torch.Tensor]) -> torch.Tensor:
        selections = []
        for path in paths:
            if path is None:
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
    
    def trace(self, nodes: torch.Tensor, parents: List[Optional[int]]) -> List[torch.Tensor]:
        path = []
        idx = len(nodes) - 1
        while idx is not None:
            path.append(nodes[idx].unsqueeze(0))
            idx = parents[idx]
        return path

    def optimize(
        self, sample_steps: int, print_freq: int = 10, debug: bool = True
    ) -> Optional[torch.Tensor]:
        self.sample_steps = sample_steps

        step = 0
        n_success = 0
        paths = [None] * self.n_trajectories
        idxs = set(range(self.n_trajectories))

        with TimerCUDA() as t:
            while step < sample_steps:
                if debug and step % print_freq == 0:
                    self.print_info(step, t.elapsed, n_success)
                    # self.visualize_trees(
                    #     traj_idx=0,
                    #     save_path=f"rrt_trees_step_{step}_planner_{self.planner_id}.png",
                    # )

                step += 1

                self.nodes_trees_1_pos, self.nodes_trees_2_pos = (
                    self.nodes_trees_2_pos,
                    self.nodes_trees_1_pos,
                )
                self.nodes_trees_1_parents, self.nodes_trees_2_parents = (
                    self.nodes_trees_2_parents,
                    self.nodes_trees_1_parents,
                )
                targets = self.sample()
                nearest_pos, nearest_idxs = self.get_nearest_nodes(
                    self.nodes_trees_1_pos, targets
                )
                extended = self.extend(
                    nearest_pos, targets, self.max_step_size, self.max_radius
                )
                n1 = self.cut(extended)
                
                self.nodes_trees_1_pos = torch.cat([self.nodes_trees_1_pos, n1.unsqueeze(1)], dim=1)
                for i in range(self.n_trajectories):
                    self.nodes_trees_1_parents[i].append(nearest_idxs[i].item())
                
                nearest_pos, nearest_idxs = self.get_nearest_nodes(
                    self.nodes_trees_2_pos, n1
                )
                extended = self.extend(
                    nearest_pos, n1, self.max_step_size, self.max_radius
                )
                n2 = self.cut(extended)

                self.nodes_trees_2_pos = torch.cat([self.nodes_trees_2_pos, n2.unsqueeze(1)], dim=1)
                for i in range(self.n_trajectories):
                    self.nodes_trees_2_parents[i].append(nearest_idxs[i].item())

                for i in list(idxs):
                    if torch.allclose(self.nodes_trees_1_pos[i][-1], self.nodes_trees_2_pos[i][-1], atol=self.eps):
                        path1 = self.trace(self.nodes_trees_1_pos[i], self.nodes_trees_1_parents[i])
                        path2 = self.trace(self.nodes_trees_2_pos[i], self.nodes_trees_2_parents[i])

                        tree_1_root = self.nodes_trees_1_pos[i][0]
                        if torch.allclose(tree_1_root, self.goal_pos, atol=self.eps):
                            path1, path2 = path2, path1

                        paths[i] = torch.cat(path1[::-1] + path2[1:], dim=0)
                        n_success += 1
                        idxs.remove(i)
    
                if n_success >= self.n_trajectories:
                    if debug:
                        self.print_info(step, t.elapsed, n_success)
                    break
                
        return self.purge_duplicates_from_trajs(paths)

    def print_info(self, step, elapsed_time, success):
        total_nodes = sum(len(self.nodes_trees_1_pos[i]) + len(self.nodes_trees_2_pos[i]) for i in range(self.n_trajectories))
        pad = len(str(self.sample_steps))
        print(
            f"| Step: {step:>{pad}}/{self.sample_steps:>{pad}} "
            f"| Time: {elapsed_time:.3f} s"
            f"| Nodes: {total_nodes} "
            f"| Success: {success}/{self.n_trajectories}"
        )

    def visualize_trees(
        self,
        traj_idx: int = 0,
        save_path: str = "rrt_trees.png",
    ):
        """Visualize the RRT trees for a specific trajectory."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for visualization")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Render environment
        from drmp.utils.visualizer import Visualizer
        vis = Visualizer(self.env, self.robot, use_extra_objects=self.use_extra_objects)
        vis._render_environment(ax, use_extra_objects=self.use_extra_objects)
        
        # Draw tree 1 (from start)
        nodes_1 = self.nodes_trees_1_pos[traj_idx].cpu().numpy()
        parents_1 = self.nodes_trees_1_parents[traj_idx]
        
        for i, parent_idx in enumerate(parents_1):
            if parent_idx is not None:
                child = nodes_1[i]
                parent = nodes_1[parent_idx]
                ax.plot([parent[0], child[0]], [parent[1], child[1]], 'b-', alpha=0.5, linewidth=0.5)
        
        ax.scatter(nodes_1[:, 0], nodes_1[:, 1], c='blue', s=10, alpha=0.6, label='Tree 1 (start)', zorder=10)
        
        # Draw tree 2 (from goal)
        nodes_2 = self.nodes_trees_2_pos[traj_idx].cpu().numpy()
        parents_2 = self.nodes_trees_2_parents[traj_idx]
        
        for i, parent_idx in enumerate(parents_2):
            if parent_idx is not None:
                child = nodes_2[i]
                parent = nodes_2[parent_idx]
                ax.plot([parent[0], child[0]], [parent[1], child[1]], 'r-', alpha=0.5, linewidth=0.5)
        
        ax.scatter(nodes_2[:, 0], nodes_2[:, 1], c='red', s=10, alpha=0.6, label='Tree 2 (goal)', zorder=10)
        
        # Draw start and goal
        start_np = self.start_pos.cpu().numpy()
        goal_np = self.goal_pos.cpu().numpy()
        ax.scatter([start_np[0]], [start_np[1]], c='cyan', s=100, marker='*', label='Start', zorder=20, edgecolors='black')
        ax.scatter([goal_np[0]], [goal_np[1]], c='magenta', s=100, marker='*', label='Goal', zorder=20, edgecolors='black')
        
        ax.legend(loc='upper right')
        ax.set_title(f'RRT-Connect Trees (Trajectory {traj_idx})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Tree visualization saved to {save_path}")   
