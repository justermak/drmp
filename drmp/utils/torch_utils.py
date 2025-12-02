from collections.abc import Mapping
from typing import List

import numpy as np
import torch


def dict_to_device(ob, device):
    if isinstance(ob, Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def to_numpy(x, dtype=np.float32, clone=False):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().astype(dtype)
        return x
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=clone)
    return np.array(x).astype(dtype)


def to_torch(x, device="cpu", dtype=torch.float, requires_grad=False, clone=False):
    if torch.is_tensor(x):
        if clone:
            x = x.clone()
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def freeze_torch_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # If the model is frozen we do not save it again, since the parameters did not change
    model.is_frozen = True


@torch.jit.script
def tensor_linspace(start: torch.Tensor, end: torch.Tensor, steps: int = 10):
    # https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


# @torch.jit.script
def batched_weighted_dot_prod(
    x: torch.Tensor, M: torch.Tensor, y: torch.Tensor, with_einsum: bool = False
):
    """
    Computes batched version of weighted dot product (distance) x.T @ M @ x
    """
    assert x.ndim >= 2
    if with_einsum:
        My = M.unsqueeze(0) @ y
        r = torch.einsum("...ij,...ij->...j", x, My)
    else:
        r = x.transpose(-2, -1) @ M.unsqueeze(0) @ x
        r = r.diagonal(dim1=-2, dim2=-1)
    return r
