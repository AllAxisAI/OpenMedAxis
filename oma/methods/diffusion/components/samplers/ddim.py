from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .base import BaseDiffusionSampler

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class DDIMSampler(BaseDiffusionSampler):
    """
    Deterministic DDIM sampler for discrete-time Gaussian diffusion.

    Based on Song et al., "Denoising Diffusion Implicit Models" (ICLR 2021).

    Key properties
    --------------
    - Deterministic at eta=0: the same starting noise always produces the same
      sample. Eliminates the per-step stochastic variance of DDPM.
    - Stochastic at eta=1: recovers approximately DDPM ancestral sampling.
    - Step-skipping: set num_inference_steps < T to run fewer reverse steps
      without retraining (e.g. 20 steps from a T=100 model).
    - Truncated start: set start_t < T-1 to begin the chain from a lower
      noise level, skipping problematic high-noise timesteps.
    - Uses the same trained model as DDPMSampler — no retraining needed.

    Update rule
    -----------
    Given x_t and model output ε_pred at timestep t, stepping to t_prev:

        x0_pred    = (x_t - √(1-ᾱ_t) · ε_pred) / √ᾱ_t
        ε_pred     = (x_t - √ᾱ_t   · x0_pred)  / √(1-ᾱ_t)
        σ_t        = η · √((1-ᾱ_{t_prev}) / (1-ᾱ_t) · (1 - ᾱ_t/ᾱ_{t_prev}))
        x_{t_prev} = √ᾱ_{t_prev} · x0_pred
                   + √(1-ᾱ_{t_prev} - σ_t²) · ε_pred
                   + σ_t · z,   z ~ N(0, I)

    At eta=0: σ_t = 0 — fully deterministic, z is never sampled.
    At eta=1: σ_t ≈ √β_t — approximately DDPM ancestral sampling.

    Notes
    -----
    - Requires a GaussianDiffusionProcess-like object (same as DDPMSampler).
    - x0_pred and ε_pred are always computed consistently from each other,
      so the update rule is valid for both epsilon- and x0-prediction models.
    - Step-skipping uses a uniform timestep grid over [0, start_t].
    """

    def __init__(
        self,
        eta: float = 0.0,
        clip_x0_pred: bool = False,
        return_intermediates: bool = False,
        num_inference_steps: Optional[int] = None,
        start_t: Optional[int] = None,
    ) -> None:
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {eta}.")

        self.eta = float(eta)
        self.clip_x0_pred = bool(clip_x0_pred)
        self.return_intermediates = bool(return_intermediates)
        self.num_inference_steps = num_inference_steps
        self.start_t = start_t

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
                "DDIMSampler requires a Gaussian-like process exposing attributes: "
                f"{missing}"
            )

    def _build_timestep_schedule(self, num_steps: int) -> List[int]:
        """
        Build the descending timestep list for the sampling loop.

        Respects start_t (maximum timestep) and num_inference_steps (count).
        Always includes 0 as the final step.

        Returns
        -------
        List[int]
            Timesteps in descending order, e.g.:
            start_t=99, num_inference_steps=5 → [99, 74, 49, 24, 0]
            start_t=80, full steps           → [80, 79, 78, ..., 0]
        """
        t_max = min(
            self.start_t if self.start_t is not None else num_steps - 1,
            num_steps - 1,
        )

        if self.num_inference_steps is not None:
            n = min(self.num_inference_steps, t_max + 1)
            raw = torch.linspace(0, t_max, n).round().long().tolist()
            timesteps = sorted(set(int(v) for v in raw), reverse=True)
        else:
            timesteps = list(range(t_max, -1, -1))

        return timesteps

    def _get_alpha_bar_prev(
        self,
        process: Any,
        t_prev: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Look up ᾱ_{t_prev}, correctly handling the t_prev = -1 convention.

        t_prev = -1 means "before timestep 0", where ᾱ = 1.0 by definition.
        This arises at the final reverse step when the chain lands at x_0.
        """
        t_prev_clamped = t_prev.clamp(min=0)
        alpha_bar = process._extract(process.alphas_cumprod, t_prev_clamped, x_shape)

        # Where t_prev < 0, override with 1.0
        is_before_start = (t_prev < 0).float().reshape(
            (t_prev.shape[0],) + (1,) * (len(x_shape) - 1)
        )
        return alpha_bar * (1.0 - is_before_start) + is_before_start

    @torch.no_grad()
    def step(
        self,
        *,
        model: Any,
        process: Any,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
        cond: Optional[Any] = None,
        method: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform one DDIM reverse step from t to t_prev.

        Parameters
        ----------
        t_prev:
            Previous timestep tensor [B]. Defaults to t - 1 when not provided.
            Pass explicitly when using step-skipping schedules.
            Use a tensor of -1s for the final step (x_prev = x0_pred).
        """
        model_pred = self.call_model(
            model=model,
            x=x_t,
            t=t,
            cond=cond,
            method=method,
            **kwargs,
        )

        state = {"xt": x_t, "t": t, "cond": cond}
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
                "DDIMSampler expected reconstruct_clean(...) to return a tensor."
            )

        if self.clip_x0_pred:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Compute ε_pred from x0_pred — valid for both epsilon- and x0-prediction
        # ε = (x_t - √ᾱ_t · x0_pred) / √(1-ᾱ_t)
        alpha_bar_t = process._extract(process.alphas_cumprod, t, x_t.shape)
        eps_pred = (x_t - alpha_bar_t.sqrt() * x0_pred) / (
            (1.0 - alpha_bar_t).clamp(min=1e-8).sqrt()
        )

        # ᾱ at t_prev (handles t_prev = -1 → 1.0)
        if t_prev is None:
            t_prev = t - 1
        alpha_bar_prev = self._get_alpha_bar_prev(process, t_prev, x_t.shape)

        # DDIM variance
        # σ_t = η · √((1-ᾱ_{prev}) / (1-ᾱ_t) · (1 - ᾱ_t/ᾱ_{prev}))
        if self.eta > 0.0:
            ratio = alpha_bar_t / alpha_bar_prev.clamp(min=1e-10)
            sigma_sq = (
                self.eta ** 2
                * (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t).clamp(min=1e-10)
                * (1.0 - ratio).clamp(min=0.0)
            ).clamp(min=0.0)
        else:
            sigma_sq = torch.zeros_like(alpha_bar_t)

        # x_{t_prev} = √ᾱ_{prev} · x0_pred + √(1-ᾱ_{prev}-σ²) · ε_pred + σ · z
        coef_x0 = alpha_bar_prev.sqrt()
        coef_eps = (1.0 - alpha_bar_prev - sigma_sq).clamp(min=0.0).sqrt()
        x_prev = coef_x0 * x0_pred + coef_eps * eps_pred

        if self.eta > 0.0:
            noise = torch.randn_like(x_t)
            nonzero_mask = (t > 0).float().reshape(
                (t.shape[0],) + (1,) * (x_t.ndim - 1)
            )
            x_prev = x_prev + nonzero_mask * sigma_sq.sqrt() * noise

        return {
            "x_prev": x_prev,
            "x0_pred": x0_pred,
            "model_pred": model_pred,
            "eps_pred": eps_pred,
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
        Run DDIM sampling over the configured timestep schedule.
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

        num_steps = int(process.betas.shape[0])
        timesteps = self._build_timestep_schedule(num_steps)

        intermediates: List[torch.Tensor] = []

        for i, t_scalar in enumerate(
            tqdm(
                timesteps,
                desc="DDIM Sampling",
                total=len(timesteps),
                leave=False,
                disable=tqdm is None or not show_progress,
            )
        ):
            t = self.make_time_tensor(
                t_value=t_scalar,
                batch_size=int(x_t.shape[0]),
                device=x_t.device,
                dtype=torch.long,
            )

            # t_prev: next lower timestep in schedule, or -1 after the last step
            t_prev_scalar = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_prev = self.make_time_tensor(
                t_value=t_prev_scalar,
                batch_size=int(x_t.shape[0]),
                device=x_t.device,
                dtype=torch.long,
            )

            out = self.step(
                model=model,
                process=process,
                x_t=x_t,
                t=t,
                t_prev=t_prev,
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
