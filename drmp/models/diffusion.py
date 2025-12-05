from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from typing import Dict
import torch
import torch.nn as nn

from drmp.models.beta_schedules import cosine_beta_schedule, exponential_beta_schedule
from drmp.models.temporal_unet import TemporalUNet
from drmp.world.robot import Robot

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


def get_models():
    return {"GaussianDiffusion": GaussianDiffusion}


class PlanningModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_context(self, robot: Robot, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def build_hard_conditions(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, context: torch.Tensor, hard_conds: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def run_inference(
        self,
        context: torch.Tensor,
        hard_conds: Dict[str, torch.Tensor],
        n_samples: int=1,
        **kwargs,
    ):
        pass


class GaussianDiffusion(PlanningModel):
    def __init__(
        self,
        n_support_points: int,
        state_dim: int,
        unet_hidden_dim: int,
        unet_dim_mults: tuple,
        unet_kernel_size: int,
        unet_resnet_block_groups: int,
        unet_random_fourier_features: bool,
        unet_learned_sin_dim: int,
        unet_attn_heads: int,
        unet_attn_head_dim: int,
        unet_context_dim: int,
        variance_schedule: str,
        n_diffusion_steps: int,
        clip_denoised: bool,
        predict_epsilon: bool,
    ):
        super().__init__()

        self.n_support_points = n_support_points
        self.state_dim = state_dim
        self.unet_hidden_dim = unet_hidden_dim
        self.unet_dim_mults = unet_dim_mults
        self.unet_kernel_size = unet_kernel_size
        self.unet_resnet_block_groups = unet_resnet_block_groups
        self.unet_random_fourier_features = unet_random_fourier_features
        self.unet_learned_sin_dim = unet_learned_sin_dim
        self.unet_attn_heads = unet_attn_heads
        self.unet_attn_head_dim = unet_attn_head_dim
        self.unet_context_dim = unet_context_dim
        self.variance_schedule = variance_schedule
        self.n_diffusion_steps = n_diffusion_steps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.model = TemporalUNet(
            input_dim=state_dim,
            hidden_dim=unet_hidden_dim,
            dim_mults=unet_dim_mults,
            kernel_size=unet_kernel_size,
            resnet_block_groups=unet_resnet_block_groups,
            random_fourier_features=unet_random_fourier_features,
            learned_sin_dim=unet_learned_sin_dim,
            attn_heads=unet_attn_heads,
            attn_head_dim=unet_attn_head_dim,
            context_dim=unet_context_dim,
        )

        if variance_schedule == "cosine":
            betas = cosine_beta_schedule(
                n_diffusion_steps, s=0.008, a_min=0, a_max=0.999
            )
        elif variance_schedule == "exponential":
            betas = exponential_beta_schedule(
                n_diffusion_steps, beta_start=1e-4, beta_end=1.0
            )
        else:
            raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = nn.MSELoss()

    def extract(self, a, t, x_shape):
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(x_shape) - 1)))

    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
            ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, context, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, context))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def apply_hard_conditioning(self, x, conditions):
        x[:, 0, :] = conditions["start_states_normalized"].clone()
        x[:, -1, :] = conditions["goal_states_normalized"].clone()
        return x

    def guide_gradient_steps(
        self,
        x,
        hard_conds,
        guide,
        n_guide_steps,
        debug=False,
    ):
        for _ in range(n_guide_steps):
            grad_scaled = guide(x)
            x = x + grad_scaled
            x = self.apply_hard_conditioning(x, hard_conds)

        return x

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape,
        hard_conds,
        context,
        n_diffusion_steps_without_noise=0,
        start_guide_steps_fraction=1.0,
        guide=None,
        n_guide_steps=1,
    ):
        device = self.betas.device
        batch_size = shape[0]
        t_start_guide = int(
            np.ceil(start_guide_steps_fraction * self.n_diffusion_steps)
        )
        x = torch.randn(shape, device=device)
        x = self.apply_hard_conditioning(x, hard_conds)

        chain = [x]

        for i in reversed(
            range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)
        ):
            t = make_timesteps(batch_size, i, device)

            t_single = t[0]
            if t_single < 0:
                t = torch.zeros_like(t)

            model_mean, _, model_log_variance = self.p_mean_variance(
                x=x, context=context, t=t
            )
            x = model_mean

            model_log_variance = self.extract(
                self.posterior_log_variance_clipped, t, x.shape
            )
            model_std = torch.exp(0.5 * model_log_variance)

            if guide is not None and t_single < t_start_guide:
                x = self.guide_gradient_steps(
                    x,
                    hard_conds=hard_conds,
                    guide=guide,
                    n_guide_steps=n_guide_steps,
                    debug=False,
                )

            # no noise when t == 0
            noise = torch.randn_like(x)
            noise[t == 0] = 0

            x = x + model_std * noise
            x = self.apply_hard_conditioning(x, hard_conds)

            chain.append(x)

        chain = torch.stack(chain, dim=1)
        return chain

    @torch.no_grad()
    def ddim_sample(
        self,
        shape,
        hard_conds,
        context,
        start_guide_steps_fraction=1.0,
        guide=None,
        n_guide_steps=1,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        device = self.betas.device
        batch_size = shape[0]
        t_start_guide = int(
            np.ceil(start_guide_steps_fraction * self.n_diffusion_steps)
        )

        total_timesteps = self.n_diffusion_steps
        sampling_timesteps = self.n_diffusion_steps // 5
        eta = 0.0

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(
            0, total_timesteps - 1, steps=sampling_timesteps + 1, device=device
        )
        times = torch.cat((torch.tensor([-1], device=device), times))
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        x = self.apply_hard_conditioning(x, hard_conds)

        chain = [x]

        for time, time_next in time_pairs:
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            model_out = self.model(x, t, context)

            x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
            pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

            if time_next < 0:
                x = x_start
                x = self.apply_hard_conditioning(x, hard_conds)
                chain.append(x)
                break

            alpha = self.extract(self.alphas_cumprod, t, x.shape)
            alpha_next = self.extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            x = x_start * alpha_next.sqrt() + c * pred_noise

            # guide gradient steps before adding noise
            if guide is not None:
                if torch.all(t_next < t_start_guide):
                    x = self.guide_gradient_steps(
                        x,
                        hard_conds=hard_conds,
                        guide=guide,
                        n_guide_steps=n_guide_steps,
                        debug=False,
                    )

            # add noise
            noise = torch.randn_like(x)
            x = x + sigma * noise
            x = self.apply_hard_conditioning(x, hard_conds)

            chain.append(x)

        return chain

    @torch.no_grad()
    def conditional_sample(
        self,
        hard_conds,
        context,
        n_diffusion_steps_without_noise=0,
        start_guide_steps_fraction=1.0,
        guide=None,
        n_guide_steps=1,
        n_samples=1,
        ddim=False,
    ):
        shape = (n_samples, self.n_support_points, self.state_dim)

        if ddim:
            return self.ddim_sample(
                shape,
                hard_conds,
                context=context,
                n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
                start_guide_steps_fraction=start_guide_steps_fraction,
                guide=guide,
                n_guide_steps=n_guide_steps,
            )

        return self.ddpm_sample(
            shape,
            hard_conds,
            context=context,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            start_guide_steps_fraction=start_guide_steps_fraction,
            guide=guide,
            n_guide_steps=n_guide_steps,
        )

    @torch.no_grad()
    def run_inference(
        self,
        context,
        hard_conds,
        n_samples=1,
        n_diffusion_steps_without_noise=0,
        start_guide_steps_fraction=1.0,
        guide=None,
        n_guide_steps=1,
        ddim=False,
    ):
        # context and hard_conds must be normalized
        context = copy(context)
        hard_conds = copy(hard_conds)

        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = v.repeat(n_samples, 1)
            hard_conds[k] = new_state

        context = context.repeat(n_samples, 1)

        # Sample from diffusion model
        chain = self.conditional_sample(
            hard_conds,
            context=context,
            n_samples=n_samples,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            start_guide_steps_fraction=start_guide_steps_fraction,
            guide=guide,
            n_guide_steps=n_guide_steps,
            ddim=ddim,
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x n_support_points x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain

        # trajs: [ (n_diffusion_steps + 1) x n_samples x n_support_points x state_dim ]
        trajs_chain_normalized = trajs_chain_normalized.permute(1, 0, 2, 3)

        return trajs_chain_normalized

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, context, t, hard_conds):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_hard_conditioning(x_noisy, hard_conds)

        x_recon = self.model(x_noisy, t, context)
        x_recon = self.apply_hard_conditioning(x_recon, hard_conds)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)

        return loss

    def loss(self, x, context, hard_conds):
        t = torch.randint(
            0, self.n_diffusion_steps, (x.shape[0],), device=x.device
        ).long()
        return self.p_losses(x, context=context, t=t, hard_conds=hard_conds)

    def build_context(self, input_dict: Dict[str, torch.Tensor]):
        context = torch.cat(
            [
                input_dict["start_states_normalized"].view(-1, self.state_dim)[:, :self.state_dim // 2],
                input_dict["goal_states_normalized"].view(-1, self.state_dim)[:, :self.state_dim // 2],
            ],
            dim=-1,
        )
        return context

    def build_hard_conditions(self, input_dict):
        hard_conds = {
            "start_states_normalized": input_dict["start_states_normalized"].view(-1, self.state_dim),
            "goal_states_normalized": input_dict["goal_states_normalized"].view(-1, self.state_dim),
        }
        return hard_conds

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        traj_normalized = input_dict["trajs_normalized"]
        context = self.build_context(input_dict)
        hard_conds = self.build_hard_conditions(input_dict)
        loss = self.loss(traj_normalized, context, hard_conds)
        loss_dict = {"diffusion_loss": loss}

        return loss_dict
