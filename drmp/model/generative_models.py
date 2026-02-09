from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drmp.dataset.dataset import TrajectoryDataset
from drmp.model.temporal_unet import TemporalUNet, TemporalUNetShortcut
from drmp.planning.guide import Guide
from drmp.visualizer import Visualizer


def get_models():
    return {
        "Diffusion": Diffusion,
        "DiffusionShortcut": DiffusionShortcut,
        "FlowMatchingShortcut": FlowMatchingShortcut,
        "Drift": Drift,
    }

def get_additional_inference_args(model_name, args):
    additional_args = {}
    if model_name in ("Diffusion", "DiffusionShortcut"):
        additional_args["n_diffusion_steps"] = args.n_diffusion_steps
        additional_args["predict_epsilon"] = args.predict_epsilon
    if model_name in ("DiffusionShortcut", "FlowMatchingShortcut"):
        additional_args["n_diffusion_steps"] = args.n_diffusion_steps
        additional_args["bootstrap_fraction"] = args.bootstrap_fraction
    if model_name == "Drift":
        additional_args["temperature"] = args.temperature
    return additional_args

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


class GenerativeModel(nn.Module, ABC):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.horizon = horizon
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.dim_mults = dim_mults
        self.kernel_size = kernel_size
        self.resnet_block_groups = resnet_block_groups
        self.positional_encoding = positional_encoding
        self.positional_encoding_dim = positional_encoding_dim
        self.attn_heads = attn_heads
        self.attn_head_dim = attn_head_dim
        self.context_dim = context_dim

        self.model = TemporalUNet(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )
    
    def build_context(self, input_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = torch.cat(
            [
                input_dict["start_pos_normalized"].view(-1, self.context_dim // 2),
                input_dict["goal_pos_normalized"].view(-1, self.context_dim // 2),
            ],
            dim=-1,
        )
        return context
        
    @abstractmethod
    def compute_loss(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        pass
    
        
class Diffusion(GenerativeModel):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        n_diffusion_steps: int,
        predict_epsilon: bool,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )
        self.n_diffusion_steps = n_diffusion_steps
        self.predict_epsilon = predict_epsilon

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


    def extract(self, a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = a.gather(-1, t)
        return out.view(-1, *((1,) * (len(shape) - 1)))

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        """
        Calculates the noise epsilon from x_t and x_0.
        Uses the formula derived from x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.
        
        Args:
            x_t: Noisy input at timestep t.
            t: Timestep indices.
            x_0: Original clean data.

        Returns:
            The standard Gaussian noise epsilon.
        """
        if self.predict_epsilon:
            return x_0
        else:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
                * x_t
                - x_0
            ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and noise epsilon.
        Formula: x_0 = (1/sqrt(alpha_bar_t)) * (x_t - sqrt(1 - alpha_bar_t) * epsilon)
        
        Args:
            x_t: Noisy input at timestep t.
            t: Timestep indices.
            noise: The noise epsilon.

        Returns:
            Estimate of x_0.
        """
        if self.predict_epsilon:
            return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
                * x_t
                - self.extract(
                    self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
                )
                * noise
            )
        else:
            return noise

    def q_posterior(
        self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        This is used during DDPM sampling to take a step backwards from t to t-1.
        
        Mean formula:
        mu_t(x_t, x_0) = (beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_0
                       + ((1 - alpha_bar_{t-1}) * sqrt(alpha_t) / (1 - alpha_bar_t)) * x_t

        Variance formula:
        sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        
        Args:
            x_0: The predicted or ground truth x_0.
            x_t: The noisy input at timestep t.
            t: The timestep indices.

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape)
            * x_0
            + self.extract(self.posterior_mean_coef2, t, x_t.shape)
            * x_t
        )
        posterior_variance = self.extract(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, x_t: torch.Tensor, context: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the p_theta(x_{t-1} | x_t) mean and variance.
        
        Args:
            x_t: Noisy input at timestep t.
            context: Conditioning context.
            t: Timestep indices.

        Returns:
            Tuple of (model_mean, posterior_variance, posterior_log_variance)
        """
        model_output = self.model(x_t, t, context)
        x_recon = self.predict_start_from_noise(x_t, t, noise=model_output)
        x_recon = torch.clamp(x_recon, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_0=x_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Diffuse the data (forward process).
        Sample from q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I).
        
        Args:
            x_0: The original clean data.
            t: The target timestep.
            noise: Optional pre-sampled noise.

        Returns:
            Diffused sample x_t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, x_0.shape)
            * x_0
            + self.extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            * noise
        )

        return sample
    
    
    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the training loss (MSE) for the diffusion model.

        1. Sample random timesteps t.
        2. Add noise to x_0 to get x_t (q_sample).
        3. Run model to predict noise (or x_0).
        4. Calculate MSE between predicted and actual noise.

        Args:
            input_dict: Dictionary containing "x" (trajectory) and context info.

        Returns:
            Dictionary with scalar "loss".
        """
        x = input_dict["x"]
        context = self.build_context(input_dict)
        
        t = torch.randint(
            0,
            self.n_diffusion_steps,
            (x.shape[0],),
            device=x.device,
        ).long()
        noise = torch.randn_like(x)

        x_noisy = self.q_sample(
            x_0=x, t=t, noise=noise
        )

        x_recon = self.model(x_noisy, t, context)

        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise)
        else:
            loss = F.mse_loss(x_recon, x)

        loss_dict = {"loss": loss}

        return loss_dict


    @torch.no_grad()
    def ddim_sample(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        eta: float,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Sample using DDIM (Denoising Diffusion Implicit Models).
        Deterministic sampling process (Probability Flow ODE) widely used for accelerated sampling.
        
        Args:
            n_samples: Number of trajectories to generate.
            context: Conditioning context.
            guide: Optional guidance function (e.g. classifier guidance).
            n_guide_steps: How many steps to apply guidance.
            t_start_guide: Timestep to start applying guidance.
            debug: Debug mode flag.
            eta: Stochasticity parameter. eta=0 is deterministic (DDIM). eta=1 is DDPM.
            n_inference_steps: Total number of inference steps (strided sampling). 
                               If None, uses n_diffusion_steps (full sampling).

        Returns:
            Generated trajectories (Batch_Size, T, Dim).
        """
        device = self.betas.device
        x_t = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )

        step_indices = torch.linspace(
            self.n_diffusion_steps - 1, 0, n_inference_steps
            ).long().flip(0).to(device)

        chain = [x_t]

        for i in reversed(range(len(step_indices))):
            t_idx = step_indices[i]
            prev_t_idx = step_indices[i - 1] if i > 0 else -1

            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            
            model_output = self.model(x_t, t, context)

            alpha_bar = self.extract(self.alphas_cumprod, t, x_t.shape)
            
            if prev_t_idx >= 0:
                prev_t = torch.full((n_samples,), prev_t_idx, device=device, dtype=torch.long)
                alpha_bar_prev = self.extract(self.alphas_cumprod, prev_t, x_t.shape)
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device) # t=0 -> prev alpha is 1

            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

            if self.predict_epsilon:
                epsilon = model_output
                pred_x0 = (
                    x_t - sqrt_one_minus_alpha_bar * epsilon
                ) / sqrt_alpha_bar
            else:
                pred_x0 = model_output
                epsilon = (
                    x_t - sqrt_alpha_bar * pred_x0
                ) / sqrt_one_minus_alpha_bar

            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Re-derive epsilon from clamped x0
            epsilon = (
                x_t - sqrt_alpha_bar * pred_x0
            ) / sqrt_one_minus_alpha_bar

            # 2. Compute DDIM step
            # Formula: x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0 + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * epsilon + sigma_t * noise
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            )

            pred_dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * epsilon
            x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + pred_dir_xt

            if (
                guide is not None
                and t_start_guide is not None
                and prev_t_idx < t_start_guide
            ):
                for _ in range(n_guide_steps):
                    x_prev = x_prev + guide(x_prev)

            if eta > 0:
                noise = torch.randn_like(x_prev)
                x_prev = x_prev + sigma_t * noise

            x_t = x_prev
            chain.append(x_t)

        chain = torch.stack(chain, dim=0)
        return chain

    @torch.no_grad()
    def ddpm_sample(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Sample using DDPM (Denoising Diffusion Probabilistic Models).
        Standard stochastic ancestral sampling.
        
        Args:
            n_samples: Number of samples.
            context: Conditioning context.
            guide: Optional guidance object.
            n_guide_steps: Number of guidance optimization steps per diffusion step.
            t_start_guide: Diffusion step to start applying guidance.
            debug: Debug flag.

        Returns:
            Generated trajectories (Batch_Size, T, Dim).
        """
        device = self.betas.device
        x_t = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )

        chain = [x_t]

        for time in reversed(range(self.n_diffusion_steps)):
            t = torch.full((n_samples,), time, device=device, dtype=torch.long)

            model_mean, _, model_log_variance = self.p_mean_variance(
                x_t=x_t, context=context, t=t
            )
            x_t = model_mean

            model_log_variance = self.extract(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
            model_std = torch.exp(0.5 * model_log_variance)

            if guide is not None and t_start_guide is not None and time <= t_start_guide:
                for _ in range(n_guide_steps):
                    x_t = x_t + guide(x_t)

            noise = torch.randn_like(x_t)
            noise[t == 0] = 0

            x_t = x_t + model_std * noise

            chain.append(x_t)

        chain = torch.stack(chain, dim=0)
        return chain

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        eta: float,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Main entry point for running inference/sampling.
        Dispatches to either DDPM or DDIM sampling based on inputs.
        
        Args:
            n_samples: Number of trajectories to generate.
            context: Context conditioning (Batch, Context_Dim) or similar.
            guide: Guidance function/object.
            n_guide_steps: Number of guidance steps.
            t_start_guide: Timestep threshold for guidance.
            n_inference_steps: Number of inference steps for DDIM. 
                               If n_inference_steps < n_diffusion_steps, DDIM is used.
            debug: debug mode.

        Returns:
            Trajectories (Batch, Horizon, Dim).
        """
        context = context.repeat(n_samples, 1)

        if n_inference_steps is not None:
            trajectories_chain_normalized = self.ddim_sample(
                n_samples=n_samples,
                context=context,
                t_start_guide=t_start_guide,
                guide=guide,
                n_guide_steps=n_guide_steps,
                debug=debug,
                n_inference_steps=n_inference_steps,
                eta=eta,
            )
        else:
            trajectories_chain_normalized = self.ddpm_sample(
                n_samples=n_samples,
                context=context,
                t_start_guide=t_start_guide,
                guide=guide,
                n_guide_steps=n_guide_steps,
                debug=debug,
            )

        return trajectories_chain_normalized



class DiffusionShortcut(Diffusion):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        n_diffusion_steps: int,
        predict_epsilon: bool,
        bootstrap_fraction: float,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
            n_diffusion_steps=n_diffusion_steps,
            predict_epsilon=predict_epsilon,
        )
        self.bootstrap_fraction = bootstrap_fraction
        self.model = TemporalUNetShortcut(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )

    def get_alpha_bar(self, t):
        t_clamped = t.clamp(min=0)
        vals = self.alphas_cumprod.gather(0, t_clamped)
        vals = torch.where(t < 0, torch.ones_like(vals), vals)
        return vals

    def ddim_step(self, x, t, dt, model_output, predict_epsilon=True):
        alpha_bar_t = self.get_alpha_bar(t)
        t_prev = t - dt
        alpha_bar_prev = self.get_alpha_bar(t_prev)
        
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - alpha_bar_t).view(-1, 1, 1)
        
        if predict_epsilon:
            noise_pred = model_output
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        else:
            pred_x0 = model_output
            noise_pred = (x - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t
            
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1. - alpha_bar_prev).view(-1, 1, 1)
        
        x_prev = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * noise_pred
        return x_prev

    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x_0 = input_dict["x"]
        context = self.build_context(input_dict)
        batch_size = x_0.shape[0]

        n_bootstrap = int(batch_size * self.bootstrap_fraction)
        n_base = batch_size - n_bootstrap

        loss_dict = {}
        total_loss = 0.0

        if n_base > 0:
            x_0_base = x_0[n_bootstrap:]
            c_base = context[n_bootstrap:]

            t = torch.randint(
                0, self.n_diffusion_steps, (n_base,), device=x_0.device
            ).long()
            noise = torch.randn_like(x_0_base)
            x_t = self.q_sample(x_0=x_0_base, t=t, noise=noise)

            dt = torch.ones_like(t)
            model_output = self.model(x_t, t, dt, c_base)

            if self.predict_epsilon:
                loss_base = F.mse_loss(model_output, noise)
            else:
                loss_base = F.mse_loss(model_output, x_0_base)

            loss_dict["loss_base"] = loss_base
            total_loss += loss_base * (n_base / batch_size)

        if n_bootstrap > 0:
            x_0_boot = x_0[:n_bootstrap]
            c_boot = context[:n_bootstrap]

            max_log2 = int(np.log2(self.n_diffusion_steps) + 1e-8)

            k_exponents = torch.randint(
                1, max_log2 + 1, (n_bootstrap,), device=x_0.device
            )
            dt = 2**k_exponents

            t = torch.randint(
                0, self.n_diffusion_steps, (n_bootstrap,), device=x_0.device
            ).long()
            
            noise = torch.randn_like(x_0_boot)
            x_t = self.q_sample(x_0=x_0_boot, t=t, noise=noise)

            # 1-step prediction (step size k)
            pred_1step_out = self.model(x_t, t, dt, c_boot)
            x_next_1step = self.ddim_step(
                x_t, t, dt, pred_1step_out, predict_epsilon=self.predict_epsilon
            )
            
            # 2-step prediction (step size k/2)
            dt_half = dt // 2
            
            # Step 1 half
            pred_half_1_out = self.model(x_t, t, dt_half, c_boot)
            x_mid = self.ddim_step(
                x_t, t, dt_half, pred_half_1_out, predict_epsilon=self.predict_epsilon
            )
            
            # Step 2 half
            t_mid = t - dt_half
            
            mask_valid = (t_mid >= 0)
            
            x_next_2step = x_mid.clone()
            
            if mask_valid.any():
                x_mid_valid = x_mid[mask_valid]
                t_mid_valid = t_mid[mask_valid]
                c_boot_valid = c_boot[mask_valid]
                dt_half_valid = dt_half[mask_valid]
                
                pred_half_2_out = self.model(x_mid_valid, t_mid_valid, dt_half_valid, c_boot_valid)
                x_final_valid = self.ddim_step(
                     x_mid_valid, t_mid_valid, dt_half_valid, pred_half_2_out, predict_epsilon=self.predict_epsilon
                )
                x_next_2step[mask_valid] = x_final_valid

            target = x_next_2step.detach()
            loss_boot = F.mse_loss(x_next_1step, target)
            loss_dict["loss_bootstrap"] = loss_boot
            total_loss += loss_boot * (n_bootstrap / batch_size)

        loss_dict["loss"] = total_loss
        return loss_dict

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)
        device = self.betas.device
        
        # Start from Noise
        trajectories = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )
        
        # Determine step size k
        # We want to go from T-1 to -1 in n_inference_steps
        step_size = self.n_diffusion_steps // n_inference_steps
        if step_size < 1:
            step_size = 1
        
        # Loop
        current_t = self.n_diffusion_steps - 1
        
        # We need integer steps.
        # e.g. T=100, shortcut=4 -> step=25.
        # 99 -> 74 -> 49 -> 24 -> -1.
        
        while current_t >= 0:
            t_tensor = torch.full((n_samples,), current_t, device=device, dtype=torch.long)
            dt_tensor = torch.full((n_samples,), step_size, device=device, dtype=torch.long)
            
            model_out = self.model(trajectories, t_tensor, dt_tensor, context)
            
            trajectories = self.ddim_step(
                trajectories, t_tensor, dt_tensor, model_out, predict_epsilon=self.predict_epsilon
            )

            current_t -= step_size
            if guide is not None and t_start_guide is not None and current_t < t_start_guide:
                for _ in range(n_guide_steps):
                    trajectories = trajectories + guide(trajectories)
            
        return trajectories.unsqueeze(0)


class FlowMatchingShortcut(GenerativeModel):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        n_diffusion_steps: int,
        bootstrap_fraction: float,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )

        self.model = TemporalUNetShortcut(
            input_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            dim_mults=self.dim_mults,
            kernel_size=self.kernel_size,
            resnet_block_groups=self.resnet_block_groups,
            positional_encoding=self.positional_encoding,
            positional_encoding_dim=self.positional_encoding_dim,
            attn_heads=self.attn_heads,
            attn_head_dim=self.attn_head_dim,
            context_dim=self.context_dim,
        )
        self.n_diffusion_steps = n_diffusion_steps
        self.bootstrap_fraction = bootstrap_fraction
        self.min_dt = 1.0 / self.n_diffusion_steps
        self.eps = 1e-5

    def compute_loss(
        self, input_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x_1 = input_dict["x"]
        x_0 = torch.randn_like(x_1)
        context = self.build_context(input_dict)

        batch_size = x_1.shape[0]
        n_bootstrap = int(batch_size * self.bootstrap_fraction)
        n_flow = batch_size - n_bootstrap

        loss_dict = {}

        total_loss = 0.0

        if n_flow > 0:
            x_1_flow = x_1[n_bootstrap:]
            x_0_flow = x_0[n_bootstrap:]
            c_flow = context[n_bootstrap:]

            t_flow = torch.rand(n_flow, device=x_1.device)
            t_flow_exp = t_flow.view(-1, 1, 1)

            x_t_flow = (
                1 - (1 - self.eps) * t_flow_exp
            ) * x_0_flow + t_flow_exp * x_1_flow

            v_target_flow = x_1_flow - (1 - self.eps) * x_0_flow

            dt_flow = torch.full((n_flow,), self.min_dt, device=x_1.device)

            v_pred_flow = self.model(x_t_flow, t_flow, dt_flow, c_flow)

            loss_flow = F.mse_loss(v_pred_flow, v_target_flow)
            loss_dict["loss_flow"] = loss_flow
            total_loss += loss_flow * (n_flow / batch_size)

        if n_bootstrap > 0:
            x_0_bootstrap = x_0[:n_bootstrap]
            x_1_bootstrap = x_1[:n_bootstrap]
            c_bootstrap = context[:n_bootstrap]

            max_log2 = int(torch.log2(torch.tensor(self.n_diffusion_steps)))
            k_exponents = torch.randint(0, max_log2, (n_bootstrap,), device=x_1.device)
            dt_bootstrap = 1.0 / (2.0**k_exponents)

            num_steps = (2**k_exponents).long()
            i_step = (torch.rand(n_bootstrap, device=x_1.device) * num_steps).long()
            t_bootstrap = i_step.float() * dt_bootstrap

            t_bootstrap_exp = t_bootstrap.view(-1, 1, 1)
            x_t_bootstrap = (
                1 - (1 - self.eps) * t_bootstrap_exp
            ) * x_0_bootstrap + t_bootstrap_exp * x_1_bootstrap

            with torch.no_grad():
                dt_half = dt_bootstrap / 2.0

                v_b1 = self.model(x_t_bootstrap, t_bootstrap, dt_half, c_bootstrap)
                x_mid = x_t_bootstrap + dt_half.view(-1, 1, 1) * v_b1

                v_b2 = self.model(x_mid, t_bootstrap + dt_half, dt_half, c_bootstrap)

                v_target_bootstrap = (v_b1 + v_b2) / 2.0

            v_pred_bootstrap = self.model(
                x_t_bootstrap, t_bootstrap, dt_bootstrap, c_bootstrap
            )

            loss_bootstrap = F.mse_loss(v_pred_bootstrap, v_target_bootstrap)
            loss_dict["loss_bootstrap"] = loss_bootstrap

            total_loss += loss_bootstrap * (n_bootstrap / batch_size)

        loss_dict["loss"] = total_loss
        return loss_dict

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        n_inference_steps: int,
        debug: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)
        device = context.device

        trajectories = torch.randn(
            (n_samples, self.horizon, self.state_dim), device=device
        )

        dt_val = 1.0 / n_inference_steps
        dt = torch.full((n_samples,), dt_val, device=device)

        current_time = 0.0
        for _ in range(n_inference_steps):
            t = torch.full((n_samples,), current_time, device=device)
            v_pred = self.model(trajectories, t, dt, context)
            trajectories = trajectories + dt_val * v_pred

            current_time += dt_val
            
            if (
                guide is not None and t_start_guide is not None
                and 1 - current_time < t_start_guide / self.n_diffusion_steps + 1e-8
            ):
                for _ in range(n_guide_steps):
                    trajectories = trajectories + guide(trajectories)

        trajectories_chain_normalized = trajectories.unsqueeze(0)

        return trajectories_chain_normalized


class Drift(GenerativeModel):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        dim_mults: tuple,
        kernel_size: int,
        resnet_block_groups: int,
        positional_encoding: str,
        positional_encoding_dim: int,
        attn_heads: int,
        attn_head_dim: int,
        context_dim: int,
        temperature: float,
    ):
        super().__init__(
            dataset=dataset,
            horizon=horizon,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            kernel_size=kernel_size,
            resnet_block_groups=resnet_block_groups,
            positional_encoding=positional_encoding,
            positional_encoding_dim=positional_encoding_dim,
            attn_heads=attn_heads,
            attn_head_dim=attn_head_dim,
            context_dim=context_dim,
        )
        self.temperature = temperature
        
    def compute_drift(self, data_generated: torch.Tensor, data_positive: torch.Tensor) -> torch.Tensor:
        """
        Compute drift field V with attention-based kernel.
        
        Args:
            data_generated: Generated samples [G, D]
            data_positive: Data samples [P, D]
        
        Returns:
            V: Drift vectors [G, D]
        """
        targets = torch.cat([data_generated, data_positive], dim=0)
        G, D = data_generated.shape

        dist = torch.cdist(data_generated, targets)
        dist[:, :G].fill_diagonal_(1e8)
        kernel = (-dist / self.temperature).exp()

        normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) # normalize along both dimensions, which we found to slightly improve performance
        normalizer = normalizer.clamp_min(1e-16).sqrt() 
        normalized_kernel = kernel # / normalizer

        pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
        pos_V = pos_coeff @ targets[G:]
        neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
        neg_V = neg_coeff @ targets[:G]

        return pos_V - neg_V

    def compute_loss(self, input_dict: Dict[str, torch.Tensor]):
        x = input_dict["x"]
        context = self.build_context(input_dict)

        B, T, D = x.shape

        z = torch.randn_like(x)
        t = torch.zeros((B,), device=x.device).long()

        x_generated = self.model(z, t, context)

        x_generated_flat = x_generated.reshape(B, -1)
        x_data_flat = x.reshape(B, -1)

        drift = self.compute_drift(x_generated_flat, x_data_flat)

        target = (x_generated_flat + drift).detach()

        loss = F.mse_loss(x_generated_flat, target)

        return {"loss": loss}

    @torch.no_grad()
    def run_inference(
        self,
        n_samples: int,
        context: torch.Tensor,
        guide: Guide,
        n_guide_steps: int,
        t_start_guide: float,
        debug: bool = False,
    ) -> torch.Tensor:
        context = context.repeat(n_samples, 1)
        device = context.device

        z = torch.randn((n_samples, self.horizon, self.state_dim), device=device)
        t = torch.zeros((n_samples,), device=device).long()

        trajectories = self.model(z, t, context)

        if guide is not None and t_start_guide is not None and t_start_guide >= 0:
            for _ in range(n_guide_steps):
                trajectories = trajectories + guide(trajectories)

        trajectories_chain_normalized = trajectories.unsqueeze(0)

        return trajectories_chain_normalized
