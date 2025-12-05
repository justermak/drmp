import random

import numpy as np
import torch


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    torch.backends.cudnn.benchmark = False


def json_to_device(ob, device):
    if isinstance(ob, dict):
        return {k: json_to_device(v, device) for k, v in ob.items()}
    elif isinstance(ob, list):
        return [json_to_device(v, device) for v in ob]
    elif torch.is_tensor(ob):
        return ob.to(device)
    else:
        return ob


def freeze_torch_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # If the model is frozen we do not save it again, since the parameters did not change
    model.is_frozen = True
