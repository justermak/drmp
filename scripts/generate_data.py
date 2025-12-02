import configargparse
import torch

from drmp.config import DEFAULT_DATA_GENERATION_ARGS
from drmp.datasets.dataset import TrajectoryDataset
from drmp.utils.seed import fix_random_seed


def run(args):
    fix_random_seed(args.seed)
    device = torch.device(args.device)
    tensor_args = {"device": device, "dtype": torch.float32}
    
    print(f"-------- GENERATING DATA --------")
    print(f"env: {args.env_name}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_trajectories per task: {args.n_trajectories}")

    dataset = TrajectoryDataset(
        datasets_dir=args.datasets_dir,
        dataset_name=args.dataset_name,
        env_name=args.env_name,
        normalizer_name=args.normalizer_name,
        robot_margin=args.robot_margin,
        cutoff_margin=args.cutoff_margin,
        tensor_args=tensor_args,
    )

    dataset.generate_data(
        n_tasks=args.n_tasks,
        n_trajectories=args.n_trajectories,
        threshold_start_goal_pos=args.threshold_start_goal_pos,
        sample_iters=args.sample_iters,
        opt_iters=args.opt_iters,
        n_support_points=args.n_support_points,
        duration=args.duration,
        val_portion=args.val_portion,
        debug=args.debug,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()

    special_args = {}

    for key, value in DEFAULT_DATA_GENERATION_ARGS.items():
        arg_name = f"--{key}"
        arg_type = type(value if value is not None else str)

        if isinstance(value, bool):
            parser.add_argument(arg_name, action="store_true", default=value)
            parser.add_argument(f"--no_{key}", dest=key, action="store_false")
        else:
            kwargs = {"type": arg_type, "default": value}
            if key in special_args:
                kwargs.update(special_args[key])
            parser.add_argument(arg_name, **kwargs)

    args = parser.parse_args()
    run(args)
