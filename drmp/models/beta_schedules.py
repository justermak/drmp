import numpy as np
import torch


def cosine_beta_schedule(
    n_diffusion_steps, s=0.008, a_min=0, a_max=0.999, dtype=torch.float32
):
    steps = n_diffusion_steps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=a_min, a_max=a_max)
    return torch.tensor(betas_clipped, dtype=dtype)


def exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    beta_start = torch.tensor(beta_start)
    beta_end = torch.tensor(beta_end)
    a = 1 / n_diffusion_steps * torch.log(beta_end / beta_start)
    return beta_start * torch.exp(a * x)
