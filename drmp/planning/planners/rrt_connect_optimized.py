from typing import Any, Dict, List, Optional, Tuple

import torch

from drmp.planning.planners.classical_planner import ClassicalPlanner
from drmp.torch_timer import TimerCUDA
from drmp.universe.environments import EnvBase
from drmp.universe.robot import RobotBase


class RRTConnectOptimized(ClassicalPlanner):
    def __init__(
        self,
        env: EnvBase,
        robot: RobotBase,
        n_trajectories: int,
        max_step_size: float,
        max_radius: float,
        n_samples: int,
        use_extra_objects: bool,
        tensor_args: Dict[str, Any],
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
        self.use_extra_objects = use_extra_objects
        self.eps = eps
        self.tensor_args = tensor_args

        # Pre-sampling buffer
        self.samples: torch.Tensor = None
        self.samples_ptr = 0
        self._initialize_samples()

    def reset(self, start_pos: torch.Tensor, goal_pos: torch.Tensor) -> None:
        """
        Initialize the planner with start and goal configurations.
        Resets the internal state of the trees.
        """
        # Ensure inputs are (1, dimmed) or (batch, dimmed)
        if start_pos.ndim == 1:
            start_pos = start_pos.unsqueeze(0)
        if goal_pos.ndim == 1:
            goal_pos = goal_pos.unsqueeze(0)
            
        # If we requested N trajectories but only provided 1 start/goal, repeat them
        if start_pos.shape[0] == 1 and self.n_trajectories > 1:
            start_pos = start_pos.repeat(self.n_trajectories, 1)
        if goal_pos.shape[0] == 1 and self.n_trajectories > 1:
            goal_pos = goal_pos.repeat(self.n_trajectories, 1)

        self.start_pos = start_pos.to(**self.tensor_args)
        self.goal_pos = goal_pos.to(**self.tensor_args)
        self.batch_size = self.start_pos.shape[0]
        self.dim = self.start_pos.shape[-1]
        
        # We don't know N_steps ahead of time, but we can pre-allocate a large buffer.
        # If we exceed this, we can re-allocate (though typically we just set max_iter).
        # Empirically, max_iter is usually passed to optimize(). 
        # Here we initialize empty containers, but they will be allocated in proper size 
        # inside optimize() once we know the max iterations.
        
        self.samples_ptr = 0

    def _initialize_samples(self) -> bool:
        samples, success = self.robot.random_collision_free_points(
            env=self.env,
            n_samples=self.n_samples,
            use_extra_objects=self.use_extra_objects,
        )
        if not success:
            print(
                "Could not find sufficient collision-free samples for RRTConnect"
            )
        self.samples = samples.to(**self.tensor_args)
        return success

    def sample(self, n_samples: int) -> torch.Tensor:
        """Batched sampling from pre-computed buffer."""
        if self.samples_ptr + n_samples > len(self.samples):
            self._initialize_samples()
            self.samples_ptr = 0

        # Return shape (n_samples, dim) -> (n_samples, dim)
        # Note: In the loop, we need (batch_size, dim), so we might need to take just `batch_size` samples if n_samples=batch_size
        samples = self.samples[self.samples_ptr : self.samples_ptr + n_samples]
        self.samples_ptr += n_samples
        return samples

    def extend_and_cut(
        self, q_near: torch.Tensor, q_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized extend + cut operation.
        Args:
            q_near: (batch, dim)
            q_target: (batch, dim)
        Returns:
            q_new: (batch, dim) - The new valid state reached 
        """
        # 1. Compute direction and raw extension
        diff = q_target - q_near
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
        
        # Normalize direction
        direction = diff / (dist + 1e-8)
        
        # Clamp distance to max_radius
        step_dist = torch.minimum(dist, torch.tensor(self.max_radius, device=dist.device))
        q_new_raw = q_near + direction * step_dist

        # 2. Check collision along the path (Interpolation)
        # Number of steps for collision checking
        n_steps = int(self.max_radius / self.max_step_size) + 2
        
        # Create interpolation factor alpha: (n_steps)
        alphas = torch.linspace(0, 1, n_steps, device=q_near.device)
        
        # Interpolate: (batch, n_steps, dim)
        # q_near: (batch, 1, dim), q_new: (batch, 1, dim)
        # alphas: (1, n_steps, 1)
        path_interpolated = q_near.unsqueeze(1) + (q_new_raw - q_near).unsqueeze(1) * alphas.view(1, -1, 1)
        
        # Check collision: (batch, n_steps)
        in_collision = self.robot.get_collision_mask(
            env=self.env, 
            points=path_interpolated, 
            on_extra=self.use_extra_objects
        )
        
        # Find first collision index
        # We want the last valid index BEFORE the first collision.
        # If no collision, we take the last index.
        
        # trick: argmax on boolean returns first True. 
        # If all False (no collision), argmax returns 0 (which is wrong), so we need to handle that.
        has_collision = in_collision.any(dim=1) # (batch,)
        first_collision_idx = in_collision.float().argmax(dim=1) # (batch,)
        
        # If has_collision is False, index should be (n_steps - 1)
        valid_idx = torch.where(has_collision, first_collision_idx - 1, torch.tensor(n_steps - 1, device=q_near.device))
        
        # Clamp to 0 (start state) at minimum to avoid indexing -1
        valid_idx = torch.clamp(valid_idx, min=0)
        
        # Gather the valid states
        # path_interpolated: (batch, n_steps, dim)
        # valid_idx: (batch,) -> needs (batch, 1, dim) expansion for gather
        batch_indices = torch.arange(q_near.shape[0], device=q_near.device)
        q_new = path_interpolated[batch_indices, valid_idx]
        
        return q_new

    def optimize(
        self, n_sampling_steps: int, print_freq: int = 50, debug: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Main RRT Connect Loop.
        Optimized for fixed memory allocation and vectorized operations.
        """
        self.n_sampling_steps = n_sampling_steps
        
        # Fixed allocation size
        # Start + Goal + (Steps * 2 trees)
        max_nodes = n_sampling_steps + 1
        
        # Tensors to store the trees
        # Shape: (batch, max_nodes, dim)
        tree_1_nodes = torch.zeros((self.batch_size, max_nodes, self.dim), **self.tensor_args)
        tree_2_nodes = torch.zeros((self.batch_size, max_nodes, self.dim), **self.tensor_args)
        
        # Parent indices
        # Shape: (batch, max_nodes)
        tree_1_parents = torch.zeros((self.batch_size, max_nodes), dtype=torch.long, device=self.start_pos.device)
        tree_2_parents = torch.zeros((self.batch_size, max_nodes), dtype=torch.long, device=self.start_pos.device)
        
        # Counts of nodes per tree in batch
        # Shape: (batch,)
        tree_1_counts = torch.ones(self.batch_size, dtype=torch.long, device=self.start_pos.device)
        tree_2_counts = torch.ones(self.batch_size, dtype=torch.long, device=self.start_pos.device)
        
        # Initialize roots
        tree_1_nodes[:, 0, :] = self.start_pos
        tree_2_nodes[:, 0, :] = self.goal_pos
        
        # Parents of roots are -1 (sentinel)
        tree_1_parents[:, 0] = -1
        tree_2_parents[:, 0] = -1
        
        # Active mask: which trajectories have not connected yet?
        active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.start_pos.device)
        
        # Connection indices (to reconstruct path later)
        # (batch,)
        connect_idx_1 = torch.zeros(self.batch_size, dtype=torch.long, device=self.start_pos.device)
        connect_idx_2 = torch.zeros(self.batch_size, dtype=torch.long, device=self.start_pos.device)
        
        # Final success mask
        success_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.start_pos.device)

        # To swap trees efficiently, we just use references
        # curr implies "growing from"
        # target implies "growing towards"
        # We swap these variables each iteration
        trees = {
            "a": {
                "nodes": tree_1_nodes, "parents": tree_1_parents, "counts": tree_1_counts, "is_start": True
            },
            "b": {
                "nodes": tree_2_nodes, "parents": tree_2_parents, "counts": tree_2_counts, "is_start": False
            }
        }
        
        curr_tree = "a"
        other_tree = "b"

        with TimerCUDA() as t:
            for step in range(n_sampling_steps):
                if not active_mask.any():
                    break
                
                if debug and step % print_freq == 0:
                     print(f"Step {step}/{n_sampling_steps} | Active: {active_mask.sum().item()} | Success: {success_mask.sum().item()}")

                # 1. Sample random targets for everyone
                # shape: (batch, dim)
                random_targets = self.sample(self.batch_size)
                
                # We only update active trajectories, but for vectorization we do dense ops usually.
                # Optimization: if strict active mask is small, we could index. 
                # But dense ops are often faster than gathering/scattering if batch is < 10k.
                
                # --- Extend Current Tree towards Random Target ---
                
                # Get current nodes buffer
                curr_nodes = trees[curr_tree]["nodes"]     # (B, N, D)
                curr_counts = trees[curr_tree]["counts"]   # (B,)
                curr_parents = trees[curr_tree]["parents"] # (B, N)
                
                # Find nearest neighbors in current tree to random sample
                # Distance: (B, N) - norm over D
                # We need to mask out invalid nodes (idx >= count)
                # Create mask: (B, N)
                node_indices = torch.arange(max_nodes, device=self.start_pos.device).unsqueeze(0).expand(self.batch_size, -1)
                valid_node_mask = node_indices < curr_counts.unsqueeze(1)
                
                # Compute distances (B, N)
                # Note: This is O(B*N). For large N, this is the bottleneck.
                # Expanding target: (B, 1, D)
                dists = torch.linalg.norm(curr_nodes - random_targets.unsqueeze(1), dim=-1)
                
                # Mask unavailable nodes with infinity
                dists_masked = torch.where(valid_node_mask & active_mask.unsqueeze(1), dists, torch.tensor(float('inf'), device=dists.device))
                
                # Get nearest
                nearest_dist, nearest_idx = torch.min(dists_masked, dim=1) # (B,)
                
                # Nodes that correspond to nearest
                nearest_node = curr_nodes[torch.arange(self.batch_size), nearest_idx] # (B, D)
                
                # Extend
                q_new = self.extend_and_cut(nearest_node, random_targets)
                
                # Add q_new to current tree
                # Only add if active AND q_new != nearest_node (meaning some progress made)
                progress_made = torch.linalg.norm(q_new - nearest_node, dim=-1) > 1e-6
                update_mask = active_mask & progress_made
                
                new_indices = curr_counts.clone() # (B,)
                
                # Update nodes tensor
                # We need to scatter the new nodes into place
                # curr_nodes[b, new_idx[b]] = q_new[b]
                # Pytorch doesn't support this direct indexing easily for batched setitem
                # Logic: trees[curr]["nodes"][b, trees[curr]["counts"], :] = q_new
                
                # Batched update using scatter is safer
                # Flatten batch and node dims? No, use indices.
                row_indices = torch.arange(self.batch_size, device=self.start_pos.device)[update_mask]
                col_indices = new_indices[update_mask]
                
                if len(row_indices) > 0:
                    curr_nodes[row_indices, col_indices] = q_new[update_mask]
                    curr_parents[row_indices, col_indices] = nearest_idx[update_mask]
                    curr_counts[update_mask] += 1
                
                # --- Extend Other Tree towards q_new (Connect Step) ---
                
                other_nodes = trees[other_tree]["nodes"]
                other_counts = trees[other_tree]["counts"]
                other_parents = trees[other_tree]["parents"]
                
                # Find nearest in other tree to q_new
                # We use q_new from ALL batches, even if some didn't extend, 
                # but valid_mask/active_mask handles correctness.
                
                dists_other = torch.linalg.norm(other_nodes - q_new.unsqueeze(1), dim=-1)
                valid_node_mask_other = node_indices < other_counts.unsqueeze(1)
                dists_other_masked = torch.where(valid_node_mask_other & active_mask.unsqueeze(1), dists_other, torch.tensor(float('inf'), device=dists.device))
                
                nearest_dist_other, nearest_idx_other = torch.min(dists_other_masked, dim=1)
                nearest_node_other = other_nodes[torch.arange(self.batch_size), nearest_idx_other]
                
                # Extend other tree towards q_new
                q_connect = self.extend_and_cut(nearest_node_other, q_new)
                
                # Add q_connect to other tree
                progress_made_other = torch.linalg.norm(q_connect - nearest_node_other, dim=-1) > 1e-6
                # Update only if previous extension happened OR we just want to greedy connect? 
                # RRT connect usually always tries to connect.
                update_mask_other = active_mask & progress_made_other
                
                row_indices_other = torch.arange(self.batch_size, device=self.start_pos.device)[update_mask_other]
                col_indices_other = other_counts[update_mask_other]
                
                if len(row_indices_other) > 0:
                    other_nodes[row_indices_other, col_indices_other] = q_connect[update_mask_other]
                    other_parents[row_indices_other, col_indices_other] = nearest_idx_other[update_mask_other]
                    other_counts[update_mask_other] += 1
                    
                # --- Check Connection ---
                # If q_connect reached q_new (dist small), we are done!
                connected = torch.linalg.norm(q_connect - q_new, dim=-1) < self.eps
                
                # Success for this batch item?
                # Must be active AND connected
                newly_finished = active_mask & connected
                
                if newly_finished.any():
                    # Record connection indices for reconstruction
                    # q_new was added at 'new_indices' in 'curr_tree'
                    # q_connect was added at 'other_counts' (pre-increment) in 'other_tree'
                    # Wait, we incremented counts. So the index is counts-1.
                    
                    finished_indices = torch.where(newly_finished)[0]
                    
                    if trees[curr_tree]["is_start"]:
                        connect_idx_1[finished_indices] = new_indices[finished_indices] # Where q_new is
                        connect_idx_2[finished_indices] = other_counts[finished_indices] - 1 # Where q_connect is
                    else:
                        # If swapped, tree 2 is current
                        connect_idx_2[finished_indices] = new_indices[finished_indices]
                        connect_idx_1[finished_indices] = other_counts[finished_indices] - 1
                        
                    active_mask[finished_indices] = False
                    success_mask[finished_indices] = True
                    
                # Swap trees
                curr_tree, other_tree = other_tree, curr_tree

        # Reconstruct paths
        # This part extracts the paths from the pre-allocated tensors
        # Output: List[torch.Tensor]
        
        # We process all, but filter by success_mask
        trajectories = []
        
        # Pull data to CPU for path reconstruction (easier to loop, and it's O(path_len))
        # Or faster: vectorize trace back? 
        # Vectorized trace back is tricky because different path lengths.
        # Given we just output a list of paths anyway, a loop over "success" indices on CPU is fine.
        
        # Move relevant pointers to CPU
        t1_nodes = tree_1_nodes.cpu()
        t1_parents = tree_1_parents.cpu()
        t2_nodes = tree_2_nodes.cpu()
        t2_parents = tree_2_parents.cpu()
        c1_idxs = connect_idx_1.cpu()
        c2_idxs = connect_idx_2.cpu()
        s_mask = success_mask.cpu()
        
        paths = []
        for i in range(self.batch_size):
            if not s_mask[i]:
                paths.append(None)
                continue
                
            # Trace tree 1
            path1 = []
            curr = c1_idxs[i].item()
            while curr != -1:
                path1.append(t1_nodes[i, curr])
                curr = t1_parents[i, curr].item()
            path1 = path1[::-1] # Reverse to get Start -> Connect
            
            # Trace tree 2
            path2 = []
            curr = c2_idxs[i].item()
            while curr != -1:
                path2.append(t2_nodes[i, curr])
                curr = t2_parents[i, curr].item()
            # Tree 2 is Goal -> Connect. Path 2 is [Connect, ..., Goal] effectively if we trace from connect.
            # But the loop traverses children->parents, i.e., Connect -> ... -> Goal.
            # So path2 contains [ConnectNode, ..., GoalNode]
            # We want Start -> Connect -> Goal.
            # path1 is [Start, ..., ConnectNode1]
            # path2 is [ConnectNode2, ..., Goal]
            # ConnectNode1 and ConnectNode2 are theoretically same (within eps)
            
            full_path = torch.stack(path1 + path2)
            paths.append(full_path.to(**self.tensor_args))
            
        # Purge duplicates (using existing method from base or similar logic)
        return self.purge_duplicates_from_trajectories(paths)

    def purge_duplicates_from_trajectories(
        self, paths: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Cleans up the paths by removing duplicate sequential points.
        Identical to the original implementation's logic.
        """
        selections = []
        for path in paths:
            if path is None:
                selections.append(None)
                continue
            if path.shape[0] <= 2:
                selections.append(path)
                continue

            diff = torch.norm(torch.diff(path, dim=-2), dim=-1)
            # Use self.eps from class
            idxs = torch.argwhere(diff > self.eps).squeeze(-1)
            selection = path[idxs]
            
            # Ensure start/goal preserved if they were caught in diff check 
            # (usually diff check keeps start because diff[0] is p[1]-p[0])
            # But we need to append the very last point since diff reduces length by 1
            
            last_point = path[-1].unsqueeze(0)
            selection = torch.cat((selection, last_point), dim=0)

            selections.append(selection)
        return selections
