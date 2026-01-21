import argparse
import os
import shutil

import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset from folder-based format to flat file format."
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
        help="Name of the source dataset (e.g., EnvDense2D_1000_100)",
    )
    parser.add_argument(
        "--target_dataset",
        type=str,
        help="Name of the target dataset. Defaults to source_dataset + '_flat'",
    )
    parser.add_argument(
        "--deduplicate_flipped",
        action="store_true",
        help="Remove trajectories that are temporally flipped duplicates (starts don't match)",
    )

    args = parser.parse_args()

    source_name = args.source_dataset
    target_name = args.target_dataset if args.target_dataset else f"{source_name}_flat"

    # Locate directories
    # Assuming script is run from project root or scripts/
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_root = os.path.join(root_dir, "datasets")

    source_dir = os.path.join(datasets_root, source_name)
    target_dir = os.path.join(datasets_root, target_name)

    # Check if paths exist, try current directory if not found in project structure
    if not os.path.exists(source_dir):
        # Try cwd relative
        cwd_datasets = os.path.join(os.getcwd(), "datasets")
        source_dir = os.path.join(cwd_datasets, source_name)
        target_dir = os.path.join(cwd_datasets, target_name)

        if not os.path.exists(source_dir):
            print(f"Source dataset not found at {source_dir}")
            return

    print(f"Source Directory: {source_dir}")
    print(f"Target Directory: {target_dir}")

    if os.path.exists(target_dir):
        print(
            f"Target directory {target_dir} already exists. Files might be overwritten."
        )
    else:
        os.makedirs(target_dir)

    # Copy config.yaml if exists
    config_src = os.path.join(source_dir, "config.yaml")
    if os.path.exists(config_src):
        shutil.copy2(config_src, os.path.join(target_dir, "config.yaml"))
        print("Copied config.yaml")

    # Iterate over directories
    # Get all subdirectories that look like task indices
    dirs = [
        d
        for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and d.isdigit()
    ]

    # Sort them to process in order (nice for tqdm)
    dirs.sort(key=lambda x: int(x))

    print(f"Found {len(dirs)} task directories.")

    count_converted = 0

    for d in tqdm(dirs, desc="Converting"):
        task_id = int(d)
        task_dir = os.path.join(source_dir, d)

        traj_free_path = os.path.join(task_dir, "trajectories-free.pt")
        traj_coll_path = os.path.join(task_dir, "trajectories-collision.pt")

        trajs = []

        t_coll = None
        t_free = None

        # Load Collision Trajectories
        if os.path.exists(traj_coll_path):
            try:
                t = torch.load(traj_coll_path, map_location="cpu")
                if t.numel() > 0:
                    t_coll = t
            except Exception as e:
                print(f"Error loading {traj_coll_path}: {e}")

        # Load Free Trajectories
        if os.path.exists(traj_free_path):
            try:
                t = torch.load(traj_free_path, map_location="cpu")
                if t.numel() > 0:
                    t_free = t
            except Exception as e:
                print(f"Error loading {traj_free_path}: {e}")

        if t_coll is None and t_free is None:
            print(f"No trajectories found for task {task_id}, skipping.")
            continue

        if args.deduplicate_flipped:
            t_free = t_free[: t_free.shape[0] // 2] if t_free is not None else None
            t_coll = t_coll[: t_coll.shape[0] // 2] if t_coll is not None else None
            if t_free is not None:
                t_free[:, 0, 2:] = 0.0
                t_free[:, -1, 2:] = 0.0
            if t_coll is not None:
                t_coll[:, 0, 2:] = 0.0
                t_coll[:, -1, 2:] = 0.0

        if t_coll is not None and t_coll.numel() > 0:
            trajs.append(t_coll)
        if t_free is not None and t_free.numel() > 0:
            trajs.append(t_free)

        if not trajs:
            continue

        # Concatenate
        all_trajs = torch.cat(trajs, dim=0)

        # Save to target
        target_path = os.path.join(target_dir, f"trajectories_{task_id}.pt")
        torch.save(all_trajs, target_path)

        # Copy image if exists
        img_src = os.path.join(task_dir, "trajectories_figure.png")
        if os.path.exists(img_src):
            shutil.copy2(
                img_src, os.path.join(target_dir, f"trajectories_figure_{task_id}.png")
            )

        count_converted += 1

    print(f"Conversion complete. Converted {count_converted} tasks.")
    print(f"New dataset is available at: {target_dir}")


if __name__ == "__main__":
    main()
