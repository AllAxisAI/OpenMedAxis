from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .base import BaseDiffusionSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class DDPMSampler(BaseDiffusionSampler):
    """
    Standard ancestral DDPM sampler for discrete-time Gaussian diffusion.

    Assumptions
    -----------
    - `process` is a GaussianDiffusionProcess-like object exposing:
        betas
        alphas
        alphas_cumprod
        alphas_cumprod_prev
        prediction_type
        _extract(...)
        _to_device(...)
    - reverse mean is computed from predicted x0
    - posterior variance uses the standard DDPM formula

    Notes
    -----
    This sampler keeps the API simple and researcher-friendly.
    It can later be extended with:
        - subset timesteps
        - return intermediates
        - clipping policies
        - guidance
        - stochasticity controls
    """

    def __init__(
        self,
        clip_x0_pred: bool = False,
        return_intermediates: bool = False,
    ) -> None:
        self.clip_x0_pred = bool(clip_x0_pred)
        self.return_intermediates = bool(return_intermediates)

    def _require_gaussian_process(self, process: Any) -> None:
        required_attrs = (
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "_extract",
        )
        missing = [name for name in required_attrs if not hasattr(process, name)]
        if missing:
            raise AttributeError(
                "DDPMSampler requires a Gaussian-like process exposing attributes: "
                f"{missing}"
            )

    def _posterior_mean(
        self,
        *,
        process: Any,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute posterior mean of q(x_{t-1} | x_t, x0_pred).
        """
        beta_t = process._extract(process.betas, t, x_t.shape)
        alpha_t = process._extract(process.alphas, t, x_t.shape)
        alpha_bar_t = process._extract(process.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev_t = process._extract(process.alphas_cumprod_prev, t, x_t.shape)

        coef1 = beta_t * torch.sqrt(alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        coef2 = (1.0 - alpha_bar_prev_t) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)

        return coef1 * x0_pred + coef2 * x_t

    def _posterior_variance(
        self,
        *,
        process: Any,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute posterior variance of q(x_{t-1} | x_t, x0_pred).
        """
        beta_t = process._extract(process.betas, t, x_t.shape)
        alpha_bar_t = process._extract(process.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev_t = process._extract(process.alphas_cumprod_prev, t, x_t.shape)

        posterior_variance = beta_t * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        return posterior_variance.clamp(min=1e-20)

    @torch.no_grad()
    def step(
        self,
        *,
        model: Any,
        process: Any,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        method: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform one reverse DDPM step.
        """
        model_pred = self.call_model(
            model=model,
            x=x_t,
            t=t,
            cond=cond,
            method=method,
            **kwargs,
        )

        state = {
            "xt": x_t,
            "t": t,
            "cond": cond,
        }
        x0_pred = self.reconstruct_clean(
            model_pred=model_pred,
            x=x_t,
            t=t,
            cond=cond,
            process=process,
            method=method,
            state=state,
        )

        if not torch.is_tensor(x0_pred):
            raise TypeError(
                "DDPMSampler expected reconstruct_clean(...) to return a tensor."
            )

        if self.clip_x0_pred:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        mean = self._posterior_mean(
            process=process,
            x_t=x_t,
            x0_pred=x0_pred,
            t=t,
        )

        variance = self._posterior_variance(
            process=process,
            x_t=x_t,
            t=t,
        )

        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().reshape((t.shape[0],) + (1,) * (x_t.ndim - 1))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise

        return {
            "x_prev": x_prev,
            "x0_pred": x0_pred,
            "model_pred": model_pred,
            "mean": mean,
            "variance": variance,
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        model: Any,
        process: Any,
        cond: Optional[Any] = None,
        shape: Optional[torch.Size] = None,
        x_init: Optional[torch.Tensor] = None,
        method: Optional[Any] = None,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Run full ancestral DDPM sampling from T-1 down to 0.
        """
        self._require_gaussian_process(process)

        device, dtype = self.get_model_device_and_dtype(model)
        x_t = self.prepare_initial_state(
            shape=shape,
            x_init=x_init,
            device=device,
            dtype=dtype,
        )

        if not torch.is_tensor(x_t):
            raise TypeError("Initial sampler state must be a torch.Tensor.")

        intermediates: List[torch.Tensor] = []
        num_steps = int(process.betas.shape[0])

        # for t_scalar in reversed(range(num_steps)):
        for t_scalar in tqdm(reversed(range(num_steps)), desc="Sampling", total=num_steps, leave=False,
                             disable=tqdm is None or not show_progress):
            
            t = self.make_time_tensor(
                t_value=t_scalar,
                batch_size=int(x_t.shape[0]),
                device=x_t.device,
                dtype=torch.long,
            )
            # print(f"Sampling step t={t_scalar} / {num_steps - 1}")

            out = self.step(
                model=model,
                process=process,
                x_t=x_t,
                t=t,
                cond=cond,
                method=method,
                **kwargs,
            )
            x_t = out["x_prev"]

            if self.return_intermediates:
                intermediates.append(x_t.detach().clone().cpu())

        if self.return_intermediates:
            return {
                "sample": x_t,
                "intermediates": intermediates,
            }

        return x_t