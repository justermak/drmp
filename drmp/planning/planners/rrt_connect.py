from typing import Any, Dict, List, Optional

import torch

from drmp.utils.torch_timer import TimerCUDA
from drmp.world.environments import EnvBase
from drmp.world.robot import Robot


class TreeNode:
    __slots__ = ("pos", "parent")

    def __init__(self, pos: torch.Tensor, parent: "TreeNode" = None):
        self.pos = pos
        self.parent = parent

    def trace(self, forward=False) -> torch.Tensor:
        sequence = []
        node = self
        while node is not None:
            sequence.append(node.pos)
            node = node.parent
        res = torch.stack(sequence) if forward else torch.stack(sequence[::-1])
        return res


class RRTConnect:
    def __init__(
        self,
        env: EnvBase,
        robot: Robot,
        step_size: float,
        n_radius: float,
        n_samples: int,
        eps: float = 1e-6,
        planner_id: int = None,
        tensor_args: Dict[str, Any] = None,
    ):
        self.tensor_args = tensor_args
        self.env = env
        self.robot = robot
        self.step_size = step_size
        self.n_radius = n_radius
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

        self.nodes_tree_1 = [TreeNode(self.start_pos)]
        self.nodes_tree_2 = [TreeNode(self.goal_pos)]
        self.nodes_tree_1_torch = self.start_pos.unsqueeze(0)
        self.nodes_tree_2_torch = self.goal_pos.unsqueeze(0)

    def _initialize_samples(self) -> bool:
        samples, success = self.env.random_collision_free_q(
            robot=self.robot, n_samples=self.n_samples
        )
        if not success:
            print(
                "Could not find sufficient collision-free start/goal pairs for RRTConnect samples"
            )
        self.samples = samples
        return success

    def sample(self) -> torch.Tensor:
        if self.samples_ptr >= len(self.samples):
            self._initialize_samples()
            self.samples_ptr = 0

        x = self.samples[self.samples_ptr]
        self.samples_ptr += 1
        return x

    def get_nearest_node(
        self, nodes: list["TreeNode"], nodes_torch: torch.Tensor, target: torch.Tensor
    ) -> "TreeNode":
        distances = torch.linalg.norm(nodes_torch - target, dim=-1)
        min_idx = torch.argmin(distances)
        return nodes[min_idx]

    def extend(
        self, q1: torch.Tensor, q2: torch.Tensor, max_step: float, max_dist: float
    ) -> List[torch.Tensor]:
        dist = torch.linalg.norm(q1 - q2, dim=-1)
        if dist > max_dist:
            q2 = q1 + (q2 - q1) * (max_dist / dist)

        alpha = torch.linspace(0, 1, int(dist / max_step) + 2).to(**self.tensor_args)
        q1 = q1.unsqueeze(0)
        q2 = q2.unsqueeze(0)
        extension = q1 + (q2 - q1) * alpha.unsqueeze(1)
        return extension

    def cut(self, sequence: torch.Tensor) -> torch.Tensor:
        in_collision = self.env.get_collision_mask(self.robot, sequence).squeeze()
        idxs_in_collision = torch.argwhere(in_collision)
        if idxs_in_collision.nelement() == 0:
            first_idx_in_collision = sequence.shape[0]
        else:
            first_idx_in_collision = idxs_in_collision[0].item()
        return sequence[:first_idx_in_collision]

    def purge_duplicates_from_traj(self, path: torch.Tensor) -> torch.Tensor:
        if path.shape[0] <= 2:
            return path

        diff = torch.norm(torch.diff(path, dim=-2), dim=-1)
        idxs = torch.argwhere(diff > self.eps).squeeze(-1)
        selection = path[idxs]

        if not torch.allclose(selection[0], path[0], atol=self.eps):
            selection = torch.cat((path[:1], selection), dim=0)
        if not torch.allclose(selection[-1], path[-1], atol=self.eps):
            selection = torch.cat((selection, path[-1:]), dim=0)
        return selection

    def optimize(
        self, sample_steps: int, traj_id: int, print_freq: int = 10, debug: bool = True
    ) -> Optional[torch.Tensor]:
        self.sample_steps = sample_steps
        self.traj_id = traj_id

        step = 0
        path = None

        with TimerCUDA() as t:
            while step < sample_steps:
                if debug and step % print_freq == 0:
                    self.print_info(step, t.elapsed, False)

                step += 1

                self.nodes_tree_1, self.nodes_tree_2 = (
                    self.nodes_tree_2,
                    self.nodes_tree_1,
                )
                self.nodes_tree_1_torch, self.nodes_tree_2_torch = (
                    self.nodes_tree_2_torch,
                    self.nodes_tree_1_torch,
                )
                target = self.sample()
                nearest = self.get_nearest_node(
                    self.nodes_tree_1, self.nodes_tree_1_torch, target
                )
                extended = self.extend(
                    nearest.pos, target, self.step_size, self.n_radius
                )
                path_segment = self.cut(extended)

                if path_segment.shape[0] == 1:
                    continue

                n1 = TreeNode(path_segment[-1], parent=nearest)
                self.nodes_tree_1.append(n1)
                self.nodes_tree_1_torch = torch.cat(
                    [self.nodes_tree_1_torch, n1.pos.unsqueeze(0)], dim=0
                )
                nearest = self.get_nearest_node(
                    self.nodes_tree_2, self.nodes_tree_2_torch, n1.pos
                )
                extended = self.extend(
                    nearest.pos, n1.pos, self.step_size, self.n_radius
                )
                path_segment = self.cut(extended)

                if path_segment.shape[0] == 1:
                    continue

                n2 = TreeNode(path_segment[-1], parent=nearest)
                self.nodes_tree_2.append(n2)
                self.nodes_tree_2_torch = torch.cat(
                    [self.nodes_tree_2_torch, n2.pos.unsqueeze(0)], dim=0
                )

                if torch.allclose(n1.pos, n2.pos, atol=self.eps):
                    tree_1_root = self.nodes_tree_1[0].pos
                    if torch.allclose(tree_1_root, self.goal_pos, atol=self.eps):
                        n1, n2 = n2, n1

                    path1 = n1.trace(forward=False)
                    path2 = n2.trace(forward=True)
                    path = torch.cat((path1, path2), dim=0)
                    break

            if path is not None and debug:
                self.print_info(step, t.elapsed, True)

        return self.purge_duplicates_from_traj(path) if path is not None else None

    def print_info(self, step, elapsed_time, success):
        prefix = (
            f"[P{self.planner_id}] [{self.traj_id}]"
            if self.planner_id is not None
            else f"[{self.traj_id}]"
        )
        total_nodes = len(self.nodes_tree_1) + len(self.nodes_tree_2)
        pad = len(str(self.sample_steps))
        print(
            f"{prefix} Step: {step:>{pad}}/{self.sample_steps:>{pad}} "
            f"| Time: {elapsed_time:.3f} s"
            f"| Nodes: {total_nodes} "
            f"| Success: {success}"
        )

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None
    ) -> "RRTConnect":
        if device is not None:
            self.tensor_args["device"] = device
        if dtype is not None:
            self.tensor_args["dtype"] = dtype

        if self.samples is not None:
            self.samples = self.samples.to(device=device, dtype=dtype)

        self.env.to(device=device, dtype=dtype)
        self.robot.to(device=device, dtype=dtype)

        return self
