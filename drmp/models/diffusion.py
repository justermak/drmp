from abc import ABC, abstractmethod
from typing import Dict, Tuple

from drmp.datasets.dataset import TrajectoryDatasetBase
from drmp.utils.trajectory_utils import fit_bsplines_to_trajectories, get_trajectories_from_bsplines
import numpy as np
import torch
import torch.nn as nn

from drmp.planning.guide import Guide
from drmp.models.temporal_unet import TemporalUNet


def get_models():
    return {
        "GaussianDiffusion": GaussianDiffusion,
        "GaussianDiffusionSplines": GaussianDiffusionSplines,
    }


def cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999):
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps + 1)
    alphas_cumprod = (
        torch.cos(((x / n_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = torch.clamp(betas, min=a_min, max=a_max)
    return betas_clipped


class DiffusionModelBase(nn.Module, ABC):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        horizon: int,
        state_dim: int,
        unet_hidden_dim: int,
        unet_dim_mults: tuple,
        unet_kernel_size: int,
        unet_resnet_block_groups: int,
        unet_positional_encoding: str,
        unet_positional_encoding_dim: int,
        unet_attn_heads: int,
        unet_attn_head_dim: int,
        unet_context_dim: int,
        n_diffusion_steps: int,
        predict_epsilon: bool,
    ):
        super().__init__()
        self.dataset = dataset
        self.horizon = horizon
        self.state_dim = state_dim
        self.unet_hidden_dim = unet_hidden_dim
        self.unet_dim_mults = unet_dim_mults
        self.unet_kernel_size = unet_kernel_size
        self.unet_resnet_block_groups = unet_resnet_block_groups
        self.unet_positional_encoding = unet_positional_encoding
        self.unet_positional_encoding_dim = unet_positional_encoding_dim
        self.unet_attn_heads = unet_attn_heads
        self.unet_attn_head_dim = unet_attn_head_dim
        self.unet_context_dim = unet_context_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.predict_epsilon = predict_epsilon

        self.model = TemporalUNet(
            input_dim=state_dim,
            hidden_dim=unet_hidden_dim,
            dim_mults=unet_dim_mults,
            kernel_size=unet_kernel_size,
            resnet_block_groups=unet_resnet_block_groups,
            positional_encoding=unet_positional_encoding,
            positional_encoding_dim=unet_positional_encoding_dim,
            attn_heads=unet_attn_heads,
            attn_head_dim=unet_attn_head_dim,
            context_dim=unet_context_dim,
        )

        betas = cosine_beta_schedule(n_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

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

    @abstractmethod
    def apply_hard_conditioning(
        self, x: torch.Tensor, hard_conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def guide_gradient_steps(
        self,
        x: torch.Tensor,
        hard_conditions: Dict[str, torch.Tensor],
        guide: Guide,
        n_guide_steps: int,
        debug: bool = False,
    ) -> torch.Tensor:
        pass

    def extract(self, a, t, x_shape) -> torch.Tensor:
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(x_shape) - 1)))

    def predict_noise_from_start(self, x_t, t, x0) -> torch.Tensor:
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

    def predict_start_from_noise(self, x_t, t, noise) -> torch.Tensor:
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

    def q_posterior(
        self, x_start, x_t, t
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, x, context, t
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, context))
        x_recon = torch.clamp(x_recon, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def build_hard_conditions(self, input_dict):
        hard_conditions = {
            "start_pos_normalized": input_dict["start_pos_normalized"],
            "goal_pos_normalized": input_dict["goal_pos_normalized"],
            "start_pos": input_dict["start_pos"],
            "goal_pos": input_dict["goal_pos"],
        }
        return hard_conditions
    
    def build_context(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = torch.cat(
            [
                input_dict["start_pos_normalized"].view(-1, self.unet_context_dim // 2),
                input_dict["goal_pos_normalized"].view(-1, self.unet_context_dim // 2),
            ],
            dim=-1,
        )
        return context

    @torch.no_grad()
    def ddpm_sample(
        self,
        n_samples: int,
        hard_conditions: Dict[str, torch.Tensor],
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
    ) -> torch.Tensor:
        device = self.betas.device
        x = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )
        x = self.apply_hard_conditioning(x, hard_conditions)

        chain = [x]

        for time in reversed(range(self.n_diffusion_steps)):
            t = torch.full((n_samples,), time, device=device, dtype=torch.long)

            model_mean, _, model_log_variance = self.p_mean_variance(
                x=x, context=context, t=t
            )
            x = model_mean

            model_log_variance = self.extract(
                self.posterior_log_variance_clipped, t, x.shape
            )
            model_std = torch.exp(0.5 * model_log_variance)

            if guide is not None and time <= t_start_guide:
                x = self.guide_gradient_steps(
                    x=x,
                    hard_conditions=hard_conditions,
                    guide=guide,
                    n_guide_steps=n_guide_steps,
                    debug=False,
                )

            noise = torch.randn_like(x)
            noise[t == 0] = 0

            x = x + model_std * noise
            x = self.apply_hard_conditioning(x, hard_conditions)

            chain.append(x)

        chain = torch.stack(chain, dim=1)
        return chain

    # Potentially broken
    # @torch.no_grad()
    # def ddim_sample(
    #     self,
    #     n_samples: int,
    #     hard_conditions: Dict[str, torch.Tensor],
    #     context: torch.Tensor,
    #     guide: Guide,
    #     n_guide_steps: int,
    #     t_start_guide: float,
    # ) -> torch.Tensor:
    #     # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
    #     device = self.betas.device

    #     total_timesteps = self.n_diffusion_steps
    #     sampling_timesteps = self.n_diffusion_steps // 5
    #     eta = 0.0

    #     # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = torch.linspace(
    #         0, total_timesteps - 1, steps=sampling_timesteps + 1, device=device
    #     )
    #     times = torch.cat((torch.tensor([-1], device=device), times))
    #     times = times.int().tolist()[::-1]
    #     time_pairs = list(
    #         zip(times[:-1], times[1:])
    #     )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     x = torch.randn((n_samples, self.horizon, self.state_dim), device=device)
    #     x = self.apply_hard_conditioning(x, hard_conditions)

    #     chain = [x]

    #     for time, time_next in time_pairs:
    #         t = torch.full((n_samples,), time, device=device, dtype=torch.long)
    #         t_next = torch.full(
    #             (n_samples,), time_next, device=device, dtype=torch.long
    #         )

    #         model_out = self.model(x, t, context)

    #         x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
    #         pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

    #         if time_next < 0:
    #             x = x_start
    #             x = self.apply_hard_conditioning(x, hard_conditions)
    #             chain.append(x)
    #             break

    #         alpha = self.extract(self.alphas_cumprod, t, x.shape)
    #         alpha_next = self.extract(self.alphas_cumprod, t_next, x.shape)

    #         sigma = (
    #             eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         )
    #         c = (1 - alpha_next - sigma**2).sqrt()

    #         x = x_start * alpha_next.sqrt() + c * pred_noise

    #         if guide is not None and time <= t_start_guide:
    #             x = self.guide_gradient_steps(
    #                 x,
    #                 hard_conditions=hard_conditions,
    #                 guide=guide,
    #                 n_guide_steps=n_guide_steps,
    #                 debug=False,
    #             )

    #         noise = torch.randn_like(x)
    #         x = x + sigma * noise
    #         x = self.apply_hard_conditioning(x, hard_conditions)

    #         chain.append(x)

    #     return chain

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        hard_conditions: Dict[str, torch.Tensor],
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        ddim: bool = False,
    ) -> torch.Tensor:
        hard_conditions = hard_conditions.copy()
        for k, v in hard_conditions.items():
            hard_conditions[k] = v.repeat(n_samples, 1)

        context = context.repeat(n_samples, 1)

        if ddim:
            trajectories_normalized = self.ddim_sample(
                n_samples=n_samples,
                hard_conditions=hard_conditions,
                context=context,
                t_start_guide=t_start_guide,
                guide=guide,
                n_guide_steps=n_guide_steps,
            )
        else:
            trajectories_normalized = self.ddpm_sample(
                n_samples=n_samples,
                hard_conditions=hard_conditions,
                context=context,
                t_start_guide=t_start_guide,
                guide=guide,
                n_guide_steps=n_guide_steps,
            )

        trajectories_chain_normalized = trajectories_normalized.permute(1, 0, 2, 3)

        return trajectories_chain_normalized

    def q_sample(self, x_start, t, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, context, t, hard_conditions) -> torch.Tensor:
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = self.apply_hard_conditioning(x_noisy, hard_conditions)

        x_recon = self.model(x_noisy, t, context)
        x_recon = self.apply_hard_conditioning(x_recon, hard_conditions)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise)
        else:
            loss = self.loss_fn(x_recon, x_start)

        return loss

    def loss(self, x, context, hard_conditions) -> torch.Tensor:
        t = torch.randint(
            0, self.n_diffusion_steps, (x.shape[0],), device=x.device
        ).long()
        return self.p_losses(x, context=context, t=t, hard_conditions=hard_conditions)


class GaussianDiffusion(DiffusionModelBase):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        horizon: int,
        state_dim: int,
        unet_hidden_dim: int,
        unet_dim_mults: tuple,
        unet_kernel_size: int,
        unet_resnet_block_groups: int,
        unet_positional_encoding: str,
        unet_positional_encoding_dim: int,
        unet_attn_heads: int,
        unet_attn_head_dim: int,
        unet_context_dim: int,
        n_diffusion_steps: int,
        predict_epsilon: bool,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            unet_hidden_dim=unet_hidden_dim,
            unet_dim_mults=unet_dim_mults,
            unet_kernel_size=unet_kernel_size,
            unet_resnet_block_groups=unet_resnet_block_groups,
            unet_positional_encoding=unet_positional_encoding,
            unet_positional_encoding_dim=unet_positional_encoding_dim,
            unet_attn_heads=unet_attn_heads,
            unet_attn_head_dim=unet_attn_head_dim,
            unet_context_dim=unet_context_dim,
            n_diffusion_steps=n_diffusion_steps,
            predict_epsilon=predict_epsilon,
        )

    def apply_hard_conditioning(self, x, conditions):
        x[:, 0, :] = conditions["start_pos_normalized"]
        x[:, -1, :] = conditions["goal_pos_normalized"]
        return x
    
    def apply_hard_conditioning_unnormalized(self, x, conditions):
        x[:, 0, :] = conditions["start_pos"]
        x[:, -1, :] = conditions["goal_pos"]
        return x

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        trajectories_normalized = input_dict["trajectories_normalized"]
        context = self.build_context(input_dict)
        hard_conditions = self.build_hard_conditions(input_dict)
        loss = self.loss(trajectories_normalized, context, hard_conditions)
        loss_dict = {"diffusion_loss": loss}

        return loss_dict
    
    def guide_gradient_steps(
        self,
        x: torch.Tensor,
        hard_conditions: Dict[str, torch.Tensor],
        guide: Guide,
        n_guide_steps: int,
        debug: bool = False,
    ) -> torch.Tensor:
        x_unnormalized = self.dataset.normalizer.unnormalize(x)
        for _ in range(n_guide_steps):
            x_unnormalized = x_unnormalized + guide(x_unnormalized)
            x_unnormalized = self.apply_hard_conditioning_unnormalized(x_unnormalized, hard_conditions)
        x = self.dataset.normalizer.normalize(x_unnormalized)
        return x


class GaussianDiffusionSplines(DiffusionModelBase):
    def __init__(
        self,
        dataset: TrajectoryDatasetBase,
        horizon: int,
        state_dim: int,
        unet_hidden_dim: int,
        unet_dim_mults: tuple,
        unet_kernel_size: int,
        unet_resnet_block_groups: int,
        unet_positional_encoding: str,
        unet_positional_encoding_dim: int,
        unet_attn_heads: int,
        unet_attn_head_dim: int,
        unet_context_dim: int,
        n_diffusion_steps: int,
        predict_epsilon: bool,
        spline_degree: int,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            unet_hidden_dim=unet_hidden_dim,
            unet_dim_mults=unet_dim_mults,
            unet_kernel_size=unet_kernel_size,
            unet_resnet_block_groups=unet_resnet_block_groups,
            unet_positional_encoding=unet_positional_encoding,
            unet_positional_encoding_dim=unet_positional_encoding_dim,
            unet_attn_heads=unet_attn_heads,
            unet_attn_head_dim=unet_attn_head_dim,
            unet_context_dim=unet_context_dim,
            n_diffusion_steps=n_diffusion_steps,
            predict_epsilon=predict_epsilon,
        )
        self.spline_degree = spline_degree

    

    def apply_hard_conditioning(self, x, conditions):
        x[:, : self.spline_degree - 1, :] = (
            conditions["start_pos_normalized"]
            .unsqueeze(1)
            .repeat(1, self.spline_degree - 1, 1)
        )
        x[:, -self.spline_degree + 1 :, :] = (
            conditions["goal_pos_normalized"]
            .unsqueeze(1)
            .repeat(1, self.spline_degree - 1, 1)
        )
        return x
    
    def apply_hard_conditioning_unnormalized_dense(self, x_dense, conditions):
        x_dense[:, 0, :] = conditions["start_pos"]
        x_dense[:, -1, :] = conditions["goal_pos"]
        return x_dense

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        control_points_normalized = input_dict["control_points_normalized"]
        context = self.build_context(input_dict)
        hard_conditions = self.build_hard_conditions(input_dict)
        loss = self.loss(control_points_normalized, context, hard_conditions)
        loss_dict = {"diffusion_loss": loss}
        return loss_dict
    
    def guide_gradient_steps(
        self,
        x: torch.Tensor,
        hard_conditions: Dict[str, torch.Tensor],
        guide: Guide,
        n_guide_steps: int,
        debug: bool = False,
    ) -> torch.Tensor:
        x_unnormalized = self.dataset.normalizer.unnormalize(x)
        x_unnormalized_dense = get_trajectories_from_bsplines(
            control_points=x_unnormalized,
            n_support_points=self.dataset.n_support_points,
            degree=self.spline_degree,
        )
        for _ in range(n_guide_steps):
            grad_scaled = guide(x=x_unnormalized_dense)
            x_unnormalized_dense = x_unnormalized_dense + grad_scaled
            x_unnormalized_dense = self.apply_hard_conditioning_unnormalized_dense(x_unnormalized_dense, hard_conditions)
        
        x_unnormalized = fit_bsplines_to_trajectories(
            trajectories=x_unnormalized_dense,
            n_control_points=self.horizon,
            degree=self.spline_degree,
        )
        x = self.dataset.normalizer.normalize(x_unnormalized)
        
        # For slow guide that I need to purge
        # for _ in range(n_guide_steps):
        #     x = x + guide(x)
        #     x = self.apply_hard_conditioning(x, hard_conditions)
        
        return x
