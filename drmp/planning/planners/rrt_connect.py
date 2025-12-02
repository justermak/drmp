from sys import path
import torch
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
from drmp.utils.trajectory_utils import (
    extend_path,
    purge_duplicates_from_traj,
    safe_path,
)
from drmp.utils.torch_timer import TimerCUDA
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


class TreeNode:
    __slots__ = ("config", "parent")  # Memory optimization

    def __init__(self, config: torch.Tensor, parent: "TreeNode" = None):
        self.config = config
        self.parent = parent

    def retrace(self) -> list["TreeNode"]:
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def render(self, ax: plt.Axes):
        if self.parent is not None:
            x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
            ax.plot([x[0], y[0]], [x[1], y[1]], color="k", linewidth=0.5)


class RRTConnect():
    def __init__(
        self,
        env: EnvBase,
        robot: Robot,
        start_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        step_size: float = 0.1,
        n_radius: float = 1.0,
        n_pre_samples: int = 10000,
        pre_samples: torch.Tensor = None,
        planner_id: int = None,
        tensor_args: Dict[str, Any] = None,
    ):
        self.tensor_args = tensor_args
        self.env = env
        self.robot = robot
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.step_size = step_size
        self.n_radius = n_radius
        self.planner_id = planner_id

        # Pre-sampling for efficiency
        self.n_pre_samples = n_pre_samples
        self.pre_samples = self._initialize_samples(pre_samples)

        # Tree structures
        self.nodes_tree_1: Optional[List[TreeNode]] = None
        self.nodes_tree_2: Optional[List[TreeNode]] = None
        self.nodes_tree_1_torch: Optional[torch.Tensor] = None
        self.nodes_tree_2_torch: Optional[torch.Tensor] = None

    def _initialize_samples(self, pre_samples: Optional[torch.Tensor]) -> torch.Tensor:
        if pre_samples is not None and len(pre_samples) > 0:
            n_needed = max(0, self.n_pre_samples - len(pre_samples))
            if n_needed > 0:
                new_samples, success = self.env.random_collision_free_q(robot=self.robot, n_samples=n_needed, max_tries=1000)
                if not success:
                    print("Could not find sufficient collision-free start/goal pairs for test tasks, try reducing the threshold, robot margin or object density")
                    return pre_samples
                return torch.cat([pre_samples, new_samples], dim=0)
            return pre_samples
        pre_samples, success = self.env.random_collision_free_q(robot=self.robot, n_samples=self.n_pre_samples, max_tries=1000)
        if not success:
            print("Could not find sufficient collision-free start/goal pairs for test tasks, try reducing the threshold, robot margin or object density")
            return torch.empty((0,), **self.tensor_args)
        return pre_samples

    def sample_collision_free(self) -> torch.Tensor:
        if len(self.pre_samples) > 0:
            idx = torch.randint(0, len(self.pre_samples), (1,))[0]
            sample = self.pre_samples[idx]
            # Remove used sample
            self.pre_samples = torch.cat(
                [self.pre_samples[:idx], self.pre_samples[idx + 1 :]]
            )
            return sample
        # Fallback: generate new sample on-the-fly
        sample, success = self.env.random_collision_free_q(robot=self.robot, n_samples=1, max_tries=1000)
        if not success:
            print("Could not find a collision-free sample on-the-fly, try reducing the threshold, robot margin or object density")
            return torch.empty((0,), **self.tensor_args)
        return sample

    def distance_fn(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(q1 - q2, dim=-1)

    def get_nearest_node(self, nodes: list["TreeNode"], nodes_torch: torch.Tensor, target: torch.Tensor) -> "TreeNode":
        distances = self.distance_fn(nodes_torch, target)
        min_idx = torch.argmin(distances)
        return nodes[min_idx]

    def extend_fn(self, q1: torch.Tensor, q2: torch.Tensor, max_step: float, max_dist: float) -> List[torch.Tensor]:
        return extend_path(
            self.distance_fn, q1, q2, max_step, max_dist, tensor_args=self.tensor_args
        )

    def collision_fn(self, q: torch.Tensor) -> torch.Tensor:
        return self.env.get_collision_mask(self.robot, q).squeeze()

    def optimize(self, sample_iters: int, timeout: float = None, print_freq: int = 10, debug: bool = False) -> Optional[torch.Tensor]:
        self.sample_iters = sample_iters
        self.timeout = timeout
        # Check start and goal validity
        if self.collision_fn(self.start_pos) or self.collision_fn(
            self.goal_pos
        ):
            print("Start or goal state is in collision")
            return None

        # Initialize trees
        self.nodes_tree_1 = [TreeNode(self.start_pos)]
        self.nodes_tree_2 = [TreeNode(self.goal_pos)]
        self.nodes_tree_1_torch = self.start_pos.unsqueeze(0)
        self.nodes_tree_2_torch = self.goal_pos.unsqueeze(0)

        iteration = 0
        success = False
        path = None

        with TimerCUDA() as t:
            while iteration < sample_iters:
                if iteration % print_freq == 0:
                    self.print_info(iteration, t.elapsed, success)

                iteration += 1

                # Always swap trees for bidirectional search
                self.nodes_tree_1, self.nodes_tree_2 = (
                    self.nodes_tree_2,
                    self.nodes_tree_1,
                )
                self.nodes_tree_1_torch, self.nodes_tree_2_torch = (
                    self.nodes_tree_2_torch,
                    self.nodes_tree_1_torch,
                )

                # Sample new target
                target = self.sample_collision_free()

                # Extend tree 1 towards target
                nearest = self.get_nearest_node(
                    self.nodes_tree_1, self.nodes_tree_1_torch, target
                )
                extended = self.extend_fn(
                    nearest.config, target, self.step_size, self.n_radius
                )
                path_segment = safe_path(extended, self.collision_fn)

                if len(path_segment) == 0:
                    continue

                # Add new node to tree 1
                n1 = TreeNode(path_segment[-1], parent=nearest)
                self.nodes_tree_1.append(n1)
                self.nodes_tree_1_torch = torch.vstack(
                    [self.nodes_tree_1_torch, n1.config]
                )

                # Try to connect tree 2 to the new node
                nearest = self.get_nearest_node(
                    self.nodes_tree_2, self.nodes_tree_2_torch, n1.config
                )
                extended = self.extend_fn(
                    nearest.config, n1.config, self.step_size, self.n_radius
                )
                path_segment = safe_path(extended, self.collision_fn)

                if len(path_segment) == 0:
                    continue

                # Add new node to tree 2
                n2 = TreeNode(path_segment[-1], parent=nearest)
                self.nodes_tree_2.append(n2)
                self.nodes_tree_2_torch = torch.vstack(
                    [self.nodes_tree_2_torch, n2.config]
                )

                # Check if trees connected
                if torch.allclose(n1.config, n2.config, atol=1e-6):
                    success = True

                    # Swap back to get correct order
                    self.nodes_tree_1, self.nodes_tree_2 = (
                        self.nodes_tree_2,
                        self.nodes_tree_1,
                    )
                    n1, n2 = n2, n1

                    # Construct path
                    path1 = [node.config for node in n1.retrace()]
                    path2 = [node.config for node in n2.retrace()]
                    path = path1[:-1] + path2[::-1]
                    break

            if path is not None and len(path) > 1:
                self.print_info(iteration, t.elapsed, success)
                
        return purge_duplicates_from_traj(path, eps=1e-6) if path is not None else None

    def print_info(self, iteration, elapsed_time, success):
        prefix = f"[P{self.planner_id}] " if self.planner_id is not None else ""
        total_nodes = (
            len(self.nodes_tree_1) + len(self.nodes_tree_2) if self.nodes_tree_1 else 0
        )
        pad = len(str(self.sample_iters))
        print(
            f"{prefix}Iteration: {iteration:>{pad}}/{self.sample_iters:>{pad}} "
            f"| Time: {elapsed_time:.3f} s"
            f"| Nodes: {total_nodes} "
            f"| Success: {success}"
        )

    def render(self, ax: plt.Axes):
        if self.nodes_tree_1:
            for node in self.nodes_tree_1:
                node.render(ax)
        if self.nodes_tree_2:
            for node in self.nodes_tree_2:
                node.render(ax)
