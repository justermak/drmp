from abc import ABC, abstractmethod
from typing import Dict, Tuple

from drmp.utils.trajectory import fit_bsplines_to_trajectories
from drmp.utils.visualizer import Visualizer
import numpy as np
import torch
import torch.nn as nn

from drmp.datasets.dataset import TrajectoryDatasetBase
from drmp.models.temporal_unet import TemporalUNet, TemporalUNetShortcut
from drmp.planning.guide import GuideBase, GuideDense


def get_models():
    return {
        "DiffusionDense": DiffusionDense,
        "DiffusionSplines": DiffusionSplines,
        "DiffusionSplinesShortcut": DiffusionSplinesShortcut,
    }


def cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999):
    trajectories = torch.linspace(0, n_diffusion_steps, n_diffusion_steps + 1)
    alphas_cumprod = (
        torch.cos(((trajectories / n_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5)
        ** 2
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
        self, trajectories: torch.Tensor, hard_conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pass

    def guide_gradient_steps(
        self,
        trajectories: torch.Tensor,
        hard_conditions: Dict[str, torch.Tensor],
        guide: GuideBase,
        n_guide_steps: int,
        debug: bool = False,
    ) -> torch.Tensor:
        for _ in range(n_guide_steps):
            trajectories = trajectories + guide(trajectories)
            trajectories = self.apply_hard_conditioning(
                trajectories, hard_conditions
            )
        return trajectories

    def extract(self, a, t, trajectories_shape) -> torch.Tensor:
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(trajectories_shape) - 1)))

    def predict_noise_from_start(self, trajectories_t, t, x0) -> torch.Tensor:
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, trajectories_t.shape)
                * trajectories_t
                - x0
            ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, trajectories_t.shape)

    def predict_start_from_noise(self, trajectories_t, t, noise) -> torch.Tensor:
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, trajectories_t.shape)
                * trajectories_t
                - self.extract(
                    self.sqrt_recipm1_alphas_cumprod, t, trajectories_t.shape
                )
                * noise
            )
        else:
            return noise

    def q_posterior(
        self, trajectories_start, trajectories_t, t
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, trajectories_t.shape)
            * trajectories_start
            + self.extract(self.posterior_mean_coef2, t, trajectories_t.shape)
            * trajectories_t
        )
        posterior_variance = self.extract(
            self.posterior_variance, t, trajectories_t.shape
        )
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, trajectories_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, trajectories, context, t
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trajectories_recon = self.predict_start_from_noise(
            trajectories, t=t, noise=self.model(trajectories, t, context)
        )
        trajectories_recon = torch.clamp(trajectories_recon, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            trajectories_start=trajectories_recon, trajectories_t=trajectories, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def build_hard_conditions(self, input_dict):
        hard_conditions = {
            "start_pos_normalized": input_dict["start_pos_normalized"],
            "goal_pos_normalized": input_dict["goal_pos_normalized"],
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
        guide: GuideBase,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
    ) -> torch.Tensor:
        device = self.betas.device
        trajectories = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )
        trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)

        chain = [trajectories]

        for time in reversed(range(self.n_diffusion_steps)):
            t = torch.full((n_samples,), time, device=device, dtype=torch.long)

            model_mean, _, model_log_variance = self.p_mean_variance(
                trajectories=trajectories, context=context, t=t
            )
            trajectories = model_mean

            model_log_variance = self.extract(
                self.posterior_log_variance_clipped, t, trajectories.shape
            )
            model_std = torch.exp(0.5 * model_log_variance)

            if guide is not None and time <= t_start_guide:
                trajectories = self.guide_gradient_steps(
                    trajectories=trajectories,
                    hard_conditions=hard_conditions,
                    guide=guide,
                    n_guide_steps=n_guide_steps,
                    debug=debug,
                )

            noise = torch.randn_like(trajectories)
            noise[t == 0] = 0

            trajectories = trajectories + model_std * noise
            trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)

            chain.append(trajectories)

        chain = torch.stack(chain, dim=1)
        return chain

    # Potentially broken
    # @torch.no_grad()
    # def ddim_sample(
    #     self,
    #     n_samples: int,
    #     hard_conditions: Dict[str, torch.Tensor],
    #     context: torch.Tensor,
    #     guide: GuideBase,
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

    #     trajectories = torch.randn((n_samples, self.horizon, self.state_dim), device=device)
    #     trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)

    #     chain = [trajectories]

    #     for time, time_next in time_pairs:
    #         t = torch.full((n_samples,), time, device=device, dtype=torch.long)
    #         t_next = torch.full(
    #             (n_samples,), time_next, device=device, dtype=torch.long
    #         )

    #         model_out = self.model(trajectories, t, context)

    #         trajectories_start = self.predict_start_from_noise(trajectories, t=t, noise=model_out)
    #         pred_noise = self.predict_noise_from_start(trajectories, t=t, x0=model_out)

    #         if time_next < 0:
    #             trajectories = trajectories_start
    #             trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)
    #             chain.append(trajectories)
    #             break

    #         alpha = self.extract(self.alphas_cumprod, t, trajectories.shape)
    #         alpha_next = self.extract(self.alphas_cumprod, t_next, trajectories.shape)

    #         sigma = (
    #             eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         )
    #         c = (1 - alpha_next - sigma**2).sqrt()

    #         trajectories = trajectories_start * alpha_next.sqrt() + c * pred_noise

    #         if guide is not None and time <= t_start_guide:
    #             trajectories = self.guide_gradient_steps(
    #                 trajectories,
    #                 hard_conditions=hard_conditions,
    #                 guide=guide,
    #                 n_guide_steps=n_guide_steps,
    #                 debug=False,
    #             )

    #         noise = torch.randn_like(trajectories)
    #         trajectories = trajectories + sigma * noise
    #         trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)

    #         chain.append(trajectories)

    #     return chain

    @torch.no_grad()
    @abstractmethod
    def run_inference(
        self,
        n_samples: int,
        hard_conditions: Dict[str, torch.Tensor],
        context: torch.Tensor,
        guide: GuideBase,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        pass

    def q_sample(self, trajectories_start, t, noise=None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(trajectories_start)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, trajectories_start.shape)
            * trajectories_start
            + self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, trajectories_start.shape
            )
            * noise
        )

        return sample

    def p_losses(self, trajectories_start, context, t) -> torch.Tensor:
        noise = torch.randn_like(trajectories_start)

        trajectories_noisy = self.q_sample(
            trajectories_start=trajectories_start, t=t, noise=noise
        )

        trajectories_recon = self.model(trajectories_noisy, t, context)

        assert noise.shape == trajectories_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(trajectories_recon, noise)
        else:
            loss = self.loss_fn(trajectories_recon, trajectories_start)

        return loss

    def loss(self, trajectories, context) -> torch.Tensor:
        t = torch.randint(
            0,
            self.n_diffusion_steps,
            (trajectories.shape[0],),
            device=trajectories.device,
        ).long()
        return self.p_losses(
            trajectories, context=context, t=t
        )


class DiffusionDense(DiffusionModelBase):
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

    def apply_hard_conditioning(self, trajectories, conditions):
        trajectories[:, 0, :] = conditions["start_pos_normalized"]
        trajectories[:, -1, :] = conditions["goal_pos_normalized"]
        return trajectories

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        trajectories_normalized = input_dict["trajectories_normalized"]
        context = self.build_context(input_dict)
        loss = self.loss(trajectories_normalized, context)
        loss_dict = {"loss": loss}

        return loss_dict
    
    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        hard_conditions: Dict[str, torch.Tensor],
        context: torch.Tensor,
        guide: GuideBase,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
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
                debug=debug,
            )

        trajectories_chain_normalized = trajectories_normalized.permute(1, 0, 2, 3)

        return trajectories_chain_normalized


class DiffusionSplines(DiffusionModelBase):
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

    def apply_hard_conditioning(self, trajectories, conditions):
        trajectories[:, : self.spline_degree - 1, :] = (
            conditions["start_pos_normalized"]
            .unsqueeze(1)
            .repeat(1, self.spline_degree - 1, 1)
        )
        trajectories[:, -self.spline_degree + 1 :, :] = (
            conditions["goal_pos_normalized"]
            .unsqueeze(1)
            .repeat(1, self.spline_degree - 1, 1)
        )
        return trajectories

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        control_points_normalized = input_dict["control_points_normalized"]
        context = self.build_context(input_dict)
        loss = self.loss(control_points_normalized, context)
        loss_dict = {"diffusion_loss": loss}
        return loss_dict
    
    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        hard_conditions: Dict[str, torch.Tensor],
        context: torch.Tensor,
        guide: GuideBase,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
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
                debug=debug,
            )

        trajectories_chain_normalized = trajectories_normalized.permute(1, 0, 2, 3)

        return trajectories_chain_normalized


class DiffusionSplinesShortcut(DiffusionSplines):
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
        bootstrap_fraction: int,
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
            spline_degree=spline_degree
        )
        
        self.model = TemporalUNetShortcut(
            input_dim=self.state_dim,
            hidden_dim=self.unet_hidden_dim,
            dim_mults=self.unet_dim_mults,
            kernel_size=self.unet_kernel_size,
            resnet_block_groups=self.unet_resnet_block_groups,
            positional_encoding=self.unet_positional_encoding,
            positional_encoding_dim=self.unet_positional_encoding_dim,
            attn_heads=self.unet_attn_heads,
            attn_head_dim=self.unet_attn_head_dim,
            context_dim=self.unet_context_dim,
        )
        
        self.bootstrap_fraction = bootstrap_fraction
        self.min_dt = 1.0 / self.n_diffusion_steps 
        self.eps = 1e-5

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        control_points_normalized = input_dict["control_points_normalized"]
        context = self.build_context(input_dict)
        
        trajectories = control_points_normalized
        batch_size = trajectories.shape[0]
        n_bootstrap = int(batch_size * self.bootstrap_fraction)
        n_flow = batch_size - n_bootstrap
        
        loss_dict = {}
        
        x_0 = torch.randn_like(trajectories)
        x_1 = trajectories
        
        total_loss = 0.0

        if n_flow > 0:
            x_1_flow = x_1[n_bootstrap:]
            x_0_flow = x_0[n_bootstrap:]
            c_flow = context[n_bootstrap:]
            
            t_flow = torch.rand(n_flow, device=trajectories.device)
            t_flow_exp = t_flow.view(-1, 1, 1)
            
            x_t_flow = (1 - (1 - self.eps) * t_flow_exp) * x_0_flow + t_flow_exp * x_1_flow
            
            v_target_flow = x_1_flow - (1 - self.eps) * x_0_flow
            
            dt_flow = torch.full((n_flow,), self.min_dt, device=trajectories.device)
            
            v_pred_flow = self.model(x_t_flow, t_flow, dt_flow, c_flow)
            
            loss_flow = self.loss_fn(v_pred_flow, v_target_flow)
            loss_dict["loss_flow"] = loss_flow
            total_loss += loss_flow * (n_flow / batch_size)
        
        if n_bootstrap > 0:
            x_0_bootstrap = x_0[:n_bootstrap]
            x_1_bootstrap = x_1[:n_bootstrap]
            c_bootstrap = context[:n_bootstrap]

            max_log2 = int(torch.log2(torch.tensor(self.n_diffusion_steps)))
            k_exponents = torch.randint(0, max_log2, (n_bootstrap,), device=trajectories.device)
            dt_bootstrap = 1.0 / (2.0 ** k_exponents)
            
            num_steps = (2 ** k_exponents).long()
            i_step = (torch.rand(n_bootstrap, device=trajectories.device) * num_steps).long()
            t_bootstrap = i_step.float() * dt_bootstrap
            
            t_bootstrap_exp = t_bootstrap.view(-1, 1, 1)
            x_t_bootstrap = (1 - (1 - self.eps) * t_bootstrap_exp) * x_0_bootstrap + t_bootstrap_exp * x_1_bootstrap
            
            with torch.no_grad():
                dt_half = dt_bootstrap / 2.0
                
                v_b1 = self.model(x_t_bootstrap, t_bootstrap, dt_half, c_bootstrap)
                x_mid = x_t_bootstrap + dt_half.view(-1, 1, 1) * v_b1
                
                v_b2 = self.model(x_mid, t_bootstrap + dt_half, dt_half, c_bootstrap)
                
                v_target_bootstrap = (v_b1 + v_b2) / 2.0

            v_pred_bootstrap = self.model(x_t_bootstrap, t_bootstrap, dt_bootstrap, c_bootstrap)
            
            loss_bootstrap = self.loss_fn(v_pred_bootstrap, v_target_bootstrap)
            loss_dict["loss_bootstrap"] = loss_bootstrap
            
            total_loss += loss_bootstrap * (n_bootstrap / batch_size)

        loss_dict["loss"] = total_loss
        return loss_dict

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        hard_conditions: Dict[str, torch.Tensor],
        context: torch.Tensor,
        guide: GuideBase,
        n_guide_steps: int,
        t_start_guide: float,
        shortcut_steps: int,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        
        hard_conditions = hard_conditions.copy()
        for k, v in hard_conditions.items():
            hard_conditions[k] = v.repeat(n_samples, 1)

        context = context.repeat(n_samples, 1)
        device = context.device
        
        trajectories = torch.randn((n_samples, self.horizon, self.state_dim), device=device)
        
        dt_val = 1.0 / shortcut_steps
        dt = torch.full((n_samples,), dt_val, device=device)
        
        current_time = 0.0
        for _ in range(shortcut_steps):
             t = torch.full((n_samples,), current_time, device=device)
             v_pred = self.model(trajectories, t, dt, context)
             trajectories = trajectories + dt_val * v_pred
             
             if guide is not None and current_time <= t_start_guide / self.n_diffusion_steps + 1e-8:
                  trajectories = self.guide_gradient_steps(
                      trajectories=trajectories,
                      hard_conditions=hard_conditions,
                      guide=guide,
                      n_guide_steps=n_guide_steps,
                      debug=debug
                  )
             
             current_time += dt_val
             
        trajectories_chain_normalized = trajectories.unsqueeze(1).permute(1, 0, 2, 3) 
        
        return trajectories_chain_normalized
