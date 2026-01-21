import os
import argparse
import torch
from torch.utils.data import random_split
from tqdm import tqdm

def recalculate_indices(dataset_dir, val_portion=0.1, seed=42):
    print(f"Processing dataset at: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # 1. Scan for trajectory files
    files = [
        f
        for f in os.listdir(dataset_dir)
        if f.startswith("trajectories_") and f.endswith(".pt") and "figure" not in f
    ]

    files_with_ids = []
    for f in files:
        try:
            # Expected format: trajectories_{id}.pt
            id_ = int(f.split("_")[1].split(".")[0])
            files_with_ids.append((id_, f))
        except Exception:
            print(f"Skipping file with unexpected name format: {f}")

    # Sort by task ID to ensure consistent order
    files_with_ids.sort(key=lambda x: x[0])
    
    if not files_with_ids:
        print("No trajectory files found.")
        return

    print(f"Found {len(files_with_ids)} trajectory files.")

    # 2. Calculate task start indices
    task_start_idxs = [0]
    current_idx = 0
    valid_task_ids = []

    print("Reading files to calculate indices...")
    for task_id, filename in tqdm(files_with_ids):
        file_path = os.path.join(dataset_dir, filename)
        try:
            # We only need the length, so map to cpu to save memory/time
            trajectories = torch.load(file_path, map_location="cpu")
            n_trajs = len(trajectories)
            
            if n_trajs > 0:
                current_idx += n_trajs
                task_start_idxs.append(current_idx)
                valid_task_ids.append(task_id)
            else:
                # If a task has 0 trajectories, we technically shouldn't include it in the index list 
                # if we want task_start_idxs[i] to correspond to valid_task_ids[i].
                # However, the original code might assume task IDs are contiguous 0..N.
                # If they are not contiguous, simply skipping might break things if code expects task_id to be index.
                # But here we are just generating indices into the huge concatenated tensor.
                # If we skip empty tasks, we just don't advance the index.
                # But we definitely need to track which task ID corresponds to which range?
                # The Dataset class concatenates everything.
                # load_data just concatenates all found files sorted by ID.
                # So if task 5 is missing, task 6 follows task 4 immediately.
                # task_start_idxs should correspond to the loaded concatenated tensor.
                pass
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    task_start_idxs_tensor = torch.tensor(task_start_idxs, dtype=torch.long)
    
    output_path = os.path.join(dataset_dir, "task_start_idxs.pt")
    torch.save(task_start_idxs_tensor, output_path)
    print(f"Saved {output_path}")

    # 3. generate train/val split
    # valid_task_ids maps index i -> task_id
    # We want to split the *indices* of the tasks (0 to n_tasks-1)
    n_tasks = len(files_with_ids) # This assumes we want to split based on the files we found
    
    # Reproducibility
    generator = torch.Generator().manual_seed(seed)
    
    # Create random split of task indices (0 to n_tasks-1)
    # Note: Dataset.load_data sorts files by ID and concatenates.
    # So task "0" in the dataset object corresponds to files_with_ids[0], "1" to files_with_ids[1], etc.
    # It does NOT necessarily correspond to the task_id integer in the filename if gaps exist.
    # But usually, task IDs are contiguous.
    
    train_size = int(n_tasks * (1 - val_portion))
    val_size = n_tasks - train_size
    
    train_tasks, val_tasks = random_split(
        range(n_tasks), [train_size, val_size], generator=generator
    )
    
    train_tasks_idxs = sorted(train_tasks.indices)
    val_tasks_idxs = sorted(val_tasks.indices)
    
    # Convert task indices to sample indices
    # task_start_idxs has length n_tasks + 1
    
    train_idxs = []
    for task_idx in train_tasks_idxs:
        start = task_start_idxs[task_idx]
        end = task_start_idxs[task_idx + 1]
        train_idxs.extend(range(start, end))

    val_idxs = []
    for task_idx in val_tasks_idxs:
        start = task_start_idxs[task_idx]
        end = task_start_idxs[task_idx + 1]
        val_idxs.extend(range(start, end))

    train_idxs_tensor = torch.tensor(train_idxs, dtype=torch.long)
    val_idxs_tensor = torch.tensor(val_idxs, dtype=torch.long)

    torch.save(train_idxs_tensor, os.path.join(dataset_dir, "train_idx.pt"))
    torch.save(val_idxs_tensor, os.path.join(dataset_dir, "val_idx.pt"))
    
    print(f"Saved train_idx.pt with {len(train_idxs)} samples")
    print(f"Saved val_idx.pt with {len(val_idxs)} samples")
    
    # Remove filtered indices if they exist as they are likely invalid now
    filtered_path = os.path.join(dataset_dir, "train_filtered_idx.pt")
    if os.path.exists(filtered_path):
        os.remove(filtered_path)
        print("Removed obsolete train_filtered_idx.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recalculate dataset indices")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset folder")
    parser.add_argument("--datasets_dir", type=str, default="datasets", help="Root datasets directory")
    parser.add_argument("--val_portion", type=float, default=0.1, help="Validation set portion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Handle relative paths properly
    if not os.path.isabs(args.datasets_dir):
        # Assuming script is run from project root or scripts folder
        # Try to find the datasets folder
        base_dir = os.getcwd()
        if os.path.basename(base_dir) == "scripts":
            base_dir = os.path.dirname(base_dir)
        full_datasets_dir = os.path.join(base_dir, args.datasets_dir)
    else:
        full_datasets_dir = args.datasets_dir

    dataset_path = os.path.join(full_datasets_dir, args.dataset_name)
    
    recalculate_indices(dataset_path, args.val_portion, args.seed)
