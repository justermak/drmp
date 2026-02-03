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

    def apply_hard_conditioning_unnormalized_dense(
        self, trajectories_dense: torch.Tensor, hard_conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        trajectories_dense[:, 0, :] = hard_conditions["start_pos"]
        trajectories_dense[:, -1, :] = hard_conditions["goal_pos"]
        return trajectories_dense

    def guide_gradient_steps(
        self,
        trajectories: torch.Tensor,
        hard_conditions: Dict[str, torch.Tensor],
        guide: GuideBase,
        n_guide_steps: int,
        debug: bool = False,
    ) -> torch.Tensor:
        if isinstance(guide, GuideDense):
            # if debug:
            #     visualizer = Visualizer(self.dataset.env, self.dataset.robot)
            trajectories_unnormalized = self.dataset.normalizer.unnormalize(
                trajectories
            )
            trajectories_dense = self.dataset.robot.get_position_interpolated(
                trajectories=trajectories_unnormalized,
                n_support_points=self.dataset.n_support_points,
            )
            # if debug:
            #     visualizer.render_scene(trajectories=trajectories_dense, save_path="debug_guide_-1.png")
            for i in range(n_guide_steps):
                trajectories_dense = trajectories_dense + guide(trajectories_dense)
                trajectories_dense = self.apply_hard_conditioning_unnormalized_dense(
                    trajectories_dense=trajectories_dense,
                    hard_conditions=hard_conditions,
                )
                # if debug:
                #     visualizer.render_scene(trajectories=trajectories_dense, save_path=f"debug_guide_{i}.png")
            trajectories_unnormalized = fit_bsplines_to_trajectories(
                trajectories=trajectories_dense,
                n_control_points=self.dataset.n_control_points,
                degree=self.dataset.robot.spline_degree,
            )
            x = self.dataset.normalizer.normalize(trajectories_unnormalized)
            return x
        else:
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

    def p_losses(self, trajectories_start, context, t, hard_conditions) -> torch.Tensor:
        noise = torch.randn_like(trajectories_start)

        trajectories_noisy = self.q_sample(
            trajectories_start=trajectories_start, t=t, noise=noise
        )
        trajectories_noisy = self.apply_hard_conditioning(
            trajectories_noisy, hard_conditions
        )

        trajectories_recon = self.model(trajectories_noisy, t, context)
        trajectories_recon = self.apply_hard_conditioning(
            trajectories_recon, hard_conditions
        )

        assert noise.shape == trajectories_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(trajectories_recon, noise)
        else:
            loss = self.loss_fn(trajectories_recon, trajectories_start)

        return loss

    def loss(self, trajectories, context, hard_conditions) -> torch.Tensor:
        t = torch.randint(
            0,
            self.n_diffusion_steps,
            (trajectories.shape[0],),
            device=trajectories.device,
        ).long()
        return self.p_losses(
            trajectories, context=context, t=t, hard_conditions=hard_conditions
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
        hard_conditions = self.build_hard_conditions(input_dict)
        loss = self.loss(trajectories_normalized, context, hard_conditions)
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
        hard_conditions = self.build_hard_conditions(input_dict)
        loss = self.loss(control_points_normalized, context, hard_conditions)
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
        n_bootstrap: int,
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
        
        self.n_bootstrap = n_bootstrap
        self.min_dt = 1.0 / self.n_diffusion_steps 

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        control_points_normalized = input_dict["control_points_normalized"]
        context = self.build_context(input_dict)
        hard_conditions = self.build_hard_conditions(input_dict)
        
        trajectories = control_points_normalized
        batch_size = trajectories.shape[0]
        n_flow = batch_size - self.n_bootstrap
        
        loss_dict = {}
        
        # x_0 (Noise)
        x_0 = torch.randn_like(trajectories)
        x_1 = trajectories
        
        # Initialize loss
        total_loss = 0.0

        # ==== 1. Flow Matching (Naive) ====
        if n_flow > 0:
            x_1_flow = x_1[self.n_bootstrap:]
            x_0_flow = x_0[self.n_bootstrap:]
            c_flow = context[self.n_bootstrap:]
            
            # Sample t uniform [0, 1]
            t_flow = torch.rand(n_flow, device=trajectories.device)
            t_flow_exp = t_flow.view(-1, 1, 1)
            
            # x_t = (1-t)x_0 + t x_1 
            x_t_flow = (1 - t_flow_exp) * x_0_flow + t_flow_exp * x_1_flow
            
            # Target v = x_1 - x_0
            v_target_flow = x_1_flow - x_0_flow
            
            # dt for flow matching (small value)
            dt_flow = torch.full((n_flow,), self.min_dt, device=trajectories.device)
            
            # Condition samples
            hard_cond_flow = {k: v[self.n_bootstrap:] for k,v in hard_conditions.items()}
            x_t_flow = self.apply_hard_conditioning(x_t_flow, hard_cond_flow)
            
            v_pred_flow = self.model(x_t_flow, t_flow, dt_flow, c_flow)
            
            loss_flow = self.loss_fn(v_pred_flow, v_target_flow)
            loss_dict["loss_flow"] = loss_flow
            total_loss += loss_flow * (n_flow / batch_size)
        
        # ==== 2. Bootstrap ====
        if self.n_bootstrap > 0:
            x_0_bst = x_0[:self.n_bootstrap]
            x_1_bst = x_1[:self.n_bootstrap]
            c_bst = context[:self.n_bootstrap]
            hard_cond_bst = {k: v[:self.n_bootstrap] for k,v in hard_conditions.items()}

            # Sample dt: powers of 2 (1, 1/2, 1/4, ...)
            # We want dt = 1/2^k. 
            # We ensure we can take at least 2 steps of size dt/2 within 1.0 effectively?
            # Actually shortcut model supports any dt=1/2^k.
            # Max steps = n_diffusion_steps.
            max_log2 = int(torch.log2(torch.tensor(self.n_diffusion_steps)))
            # exponents k in [0, max_log2-1]
            k_exponents = torch.randint(0, max_log2, (self.n_bootstrap,), device=trajectories.device)
            dt_bst = 1.0 / (2.0 ** k_exponents)
            
            # Sample t aligned with dt
            # t = i * dt. i chosen uniformly from valid steps
            num_steps = (1.0 / dt_bst).long()
            # If num_steps is 1 (dt=1), i=0.
            # If num_steps is 2 (dt=0.5), i in {0, 1}.
            # We use floor(rand * num_steps)
            i_step = (torch.rand(self.n_bootstrap, device=trajectories.device) * num_steps).long()
            t_bst = i_step.float() * dt_bst
            
            t_bst_exp = t_bst.view(-1, 1, 1)
            
            # x_t = (1-t)x_0 + t x_1 (Flow interpolation)
            x_t_bst = (1 - t_bst_exp) * x_0_bst + t_bst_exp * x_1_bst
            x_t_bst = self.apply_hard_conditioning(x_t_bst, hard_cond_bst)
            
            # Compute Target (Double step)
            # Use no_grad for teacher
            with torch.no_grad():
                dt_half = dt_bst / 2.0
                
                # Step 1
                v_b1 = self.model(x_t_bst, t_bst, dt_half, c_bst)
                x_mid = x_t_bst + dt_half.view(-1, 1, 1) * v_b1
                x_mid = self.apply_hard_conditioning(x_mid, hard_cond_bst)
                
                # Step 2
                v_b2 = self.model(x_mid, t_bst + dt_half, dt_half, c_bst)
                
                # Target
                v_target_bst = (v_b1 + v_b2) / 2.0

            # Predict student
            v_pred_bst = self.model(x_t_bst, t_bst, dt_bst, c_bst)
            
            loss_bootstrap = self.loss_fn(v_pred_bst, v_target_bst)
            loss_dict["loss_bootstrap"] = loss_bootstrap
            
            total_loss += loss_bootstrap * (self.n_bootstrap / batch_size)

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
        
        # x_0 (Noise)
        trajectories = torch.randn((n_samples, self.horizon, self.state_dim), device=device)
        trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)
        
        dt_val = 1.0 / shortcut_steps
        dt = torch.full((n_samples,), dt_val, device=device)
        
        current_time = 0.0
        for _ in range(shortcut_steps):
             t = torch.full((n_samples,), current_time, device=device)
             
             # Predict v
             v_pred = self.model(trajectories, t, dt, context)
             
             # Step: x_{t+dt} = x_t + dt * v
             trajectories = trajectories + dt_val * v_pred
             trajectories = self.apply_hard_conditioning(trajectories, hard_conditions)
             
             if guide is not None and current_time <= t_start_guide / self.n_diffusion_steps + 1e-8:
                  trajectories = self.guide_gradient_steps(
                      trajectories=trajectories,
                      hard_conditions=hard_conditions,
                      guide=guide,
                      n_guide_steps=n_guide_steps,
                      debug=debug
                  )
             
             current_time += dt_val
             
        # Add singleton dimension for chain (only one step in chain effectively, or final result)
        trajectories_chain_normalized = trajectories.unsqueeze(1).permute(1, 0, 2, 3) 
        
        return trajectories_chain_normalized
