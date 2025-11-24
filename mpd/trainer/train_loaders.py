import os

import torch
from torch.utils.data import DataLoader, random_split

from mpd import models, losses, datasets
from mpd.utils.torch_utils import freeze_torch_model_params


def get_model(
    model_class=None,
    checkpoint_path=None,
    freeze_loaded_model=False,
    tensor_args=None,
    **kwargs,
):
    if checkpoint_path is not None:
        model = torch.load(checkpoint_path)
        if freeze_loaded_model:
            freeze_torch_model_params(model)
    else:
        ModelClass = getattr(models, model_class)
        model = ModelClass(**kwargs).to(tensor_args["device"])

    return model


def build_module(model_class=None, submodules=None, **kwargs):
    if submodules is not None:
        for key, value in submodules.items():
            kwargs[key] = build_module(**value)

    Model = getattr(models, model_class)
    model = Model(**kwargs)

    return model


def get_dataset(
    dataset_class=None,
    dataset_dir=None,
    batch_size=2,
    val_set_size=0.05,
    logs_dir=None,
    save_indices=False,
    **kwargs,
):
    DatasetClass = getattr(datasets, dataset_class)
    print("\n---------------Loading data")
    full_dataset = DatasetClass(dataset_subdir=dataset_dir, **kwargs)
    print(full_dataset)

    # split into train and validation
    train_subset, val_subset = random_split(
        full_dataset, [1 - val_set_size, val_set_size]
    )
    train_dataloader = DataLoader(train_subset, batch_size=batch_size)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)

    if save_indices:
        # save the indices of training and validation sets (for later evaluation)
        os.makedirs(logs_dir, exist_ok=True)
        torch.save(
            train_subset.indices, os.path.join(logs_dir, f"train_subset_indices.pt")
        )
        torch.save(
            val_subset.indices, os.path.join(logs_dir, f"val_subset_indices.pt")
        )

    return train_subset, train_dataloader, val_subset, val_dataloader
