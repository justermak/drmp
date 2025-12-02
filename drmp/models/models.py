from abc import ABC, abstractmethod
from copy import copy

import einops
import numpy as np
import torch
import torch.nn as nn

from drmp.models.beta_schedules import cosine_beta_schedule, exponential_beta_schedule
from drmp.models.temporal_unet import TemporalUnet


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


def get_models():
    return {"GaussianDiffusion": GaussianDiffusion}


class PlanningModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_context(self, input_dict):
        pass

    @abstractmethod
    def build_hard_conds(self, input_dict):
        pass

    @abstractmethod
    def compute_loss(self, x, context, hard_conds):
        pass
    
    @abstractmethod
    def run_inference(
        self,
        context,
        hard_conds,
        n_samples=1,
        **kwargs,
    ):
        pass


class GaussianDiffusion(PlanningModel):
    def __init__(
        self,
        n_support_points=None,
        state_dim=None,
        unet_input_dim=32,
        unet_dim_mults=(1, 2, 4, 8),
        time_emb_dim=32,
        self_attention=False,
        conditioning_embed_dim=4,
        conditioning_type=None,
        attention_num_heads=2,
        attention_dim_head=32,
        variance_schedule="exponential",
        n_diffusion_steps=100,
        clip_denoised=True,
        predict_epsilon=False,
    ):
        super().__init__()

        self.n_support_points = n_support_points
        self.state_dim = state_dim
        self.unet_input_dim = unet_input_dim
        self.unet_dim_mults = unet_dim_mults
        self.time_emb_dim = time_emb_dim
        self.self_attention = self_attention
        self.conditioning_embed_dim = conditioning_embed_dim
        self.conditioning_type = conditioning_type
        self.attention_num_heads = attention_num_heads
        self.attention_dim_head = attention_dim_head
        self.variance_schedule = variance_schedule
        self.n_diffusion_steps = n_diffusion_steps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.model = TemporalUnet(
            state_dim=state_dim,
            n_support_points=n_support_points,
            unet_input_dim=unet_input_dim,
            unet_dim_mults=unet_dim_mults,
            time_emb_dim=time_emb_dim,
            self_attention=self_attention,
            conditioning_embed_dim=conditioning_embed_dim,
            conditioning_type=conditioning_type,
            attention_num_heads=attention_num_heads,
            attention_dim_head=attention_dim_head,
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
        n_guide_steps=1,
        scale_grad_by_std=False,
        model_var=None,
        debug=False,
    ):
        for _ in range(n_guide_steps):
            grad_scaled = guide(x)

            if scale_grad_by_std:
                grad_scaled = model_var * grad_scaled

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
        scale_grad_by_std=False,
    ):
        device = self.betas.device
        batch_size = shape[0]
        t_start_guide = int(np.ceil(start_guide_steps_fraction * self.n_diffusion_steps))
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
            model_var = torch.exp(model_log_variance)

            if guide is not None and t_single < t_start_guide:
                x = self.guide_gradient_steps(
                    x,
                    hard_conds=hard_conds,
                    guide=guide,
                    n_guide_steps=n_guide_steps,
                    scale_grad_by_std=scale_grad_by_std,
                    model_var=model_var,
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
        scale_grad_by_std=False,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        device = self.betas.device
        batch_size = shape[0]
        t_start_guide = int(np.ceil(start_guide_steps_fraction * self.n_diffusion_steps))
        
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
                        x, hard_conds=hard_conds, guide=guide, n_guide_steps=n_guide_steps,
                        scale_grad_by_std=scale_grad_by_std, model_var=None, debug=False,
                    )

            # add noise
            noise = torch.randn_like(x)
            x = x + sigma * noise
            x = self.apply_hard_conditioning(x, hard_conds)

            chain.append(x)

        return chain

    @torch.no_grad()
    def conditional_sample(self, hard_conds,
        context,
        n_diffusion_steps_without_noise=0,
        start_guide_steps_fraction=1.0,
        guide=None,
        n_guide_steps=1,
        scale_grad_by_std=False,
        n_samples=1,
        ddim=False,
    ):
        shape = (n_samples, self.n_support_points, self.state_dim)

        if ddim:
            return self.ddim_sample(shape, hard_conds, context=context,
                n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
                start_guide_steps_fraction=start_guide_steps_fraction,
                guide=guide,
                n_guide_steps=n_guide_steps,
                scale_grad_by_std=scale_grad_by_std,
            )

        return self.ddpm_sample(shape, hard_conds, context=context,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            start_guide_steps_fraction=start_guide_steps_fraction,
            guide=guide,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
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
        scale_grad_by_std=False,
        ddim=False,
    ):
        # context and hard_conds must be normalized
        context = copy(context)
        hard_conds = copy(hard_conds)

        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, "d -> b d", b=n_samples)
            hard_conds[k] = new_state

        context = einops.repeat(context, "d -> b d", b=n_samples)

        # Sample from diffusion model
        chain = self.conditional_sample(
            hard_conds,
            context=context,
            n_samples=n_samples,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            start_guide_steps_fraction=start_guide_steps_fraction,
            guide=guide,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            ddim=ddim,
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x n_support_points x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain

        # trajs: [ (n_diffusion_steps + 1) x n_samples x n_support_points x state_dim ]
        trajs_chain_normalized = einops.rearrange(
            trajs_chain_normalized, "b diffsteps h d -> diffsteps b h d"
        )

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

    def build_context(self, input_dict):
        context = torch.cat(
            [
                input_dict["start_states_normalized"],
                input_dict["goal_states_normalized"],
            ],
            dim=-1,
        )
        return context

    def build_hard_conds(self, input_dict):
        hard_conds = {
            "start_states_normalized": input_dict["start_states_normalized"],
            "goal_states_normalized": input_dict["goal_states_normalized"],
        }
        return hard_conds

    def compute_loss(self, input_dict):
        traj_normalized = input_dict["trajs_normalized"]
        context = self.build_context(input_dict)
        hard_conds = self.build_hard_conds(input_dict)
        loss = self.loss(traj_normalized, context, hard_conds)
        loss_dict = {"diffusion_loss": loss}

        return loss_dict
