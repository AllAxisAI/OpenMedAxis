from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDiffusionProcess


def _linear_beta_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def _cosine_beta_schedule(
    num_steps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """
    Cosine schedule inspired by Nichol & Dhariwal.

    Returns betas for discrete DDPM-style training.
    """
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-8, max=max_beta)


class GaussianDiffusionProcess(BaseDiffusionProcess, nn.Module):
    """
    Standard discrete-time Gaussian diffusion process.

    This process owns the forward corruption path:
        q(x_t | x_0)

    It is intentionally method-agnostic and can be reused by different
    diffusion methods as long as they follow the OMA diffusion state style.

    Supported prediction styles for x0 reconstruction:
        - epsilon
        - x0
        - v

    Inherits nn.Module so that schedule buffers move automatically when the
    parent method is moved to a device (e.g. method.to("cuda")).
    Buffers are registered with persistent=False — they are recomputable from
    constructor arguments and do not need to be saved in checkpoints.

    Notes
    -----
    - `t` is expected to be an integer tensor of shape [B].
    - The process returns a rich dict to support objectives, losses, and logging.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
        prediction_type: str = "epsilon",
        clip_x0_pred: bool = False,
        schedule_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        self.num_steps = int(num_steps)
        self.schedule = schedule.lower()
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.cosine_s = float(cosine_s)
        self.prediction_type = prediction_type.lower()
        self.clip_x0_pred = bool(clip_x0_pred)
        self.schedule_fn = schedule_fn

        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0.")

        if schedule_fn is None and self.schedule not in {"linear", "cosine"}:
            raise ValueError(
                f"Unsupported schedule '{schedule}'. "
                "Expected one of ['linear', 'cosine'], or pass a custom schedule_fn."
            )

        if self.prediction_type not in {"epsilon", "x0", "v"}:
            raise ValueError(
                f"Unsupported prediction_type '{prediction_type}'. "
                "Expected one of ['epsilon', 'x0', 'v']."
            )

        betas = self._build_betas()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas",                          betas,                               persistent=False)
        self.register_buffer("alphas",                         alphas,                              persistent=False)
        self.register_buffer("alphas_cumprod",                 alphas_cumprod,                      persistent=False)
        self.register_buffer("alphas_cumprod_prev",            alphas_cumprod_prev,                 persistent=False)
        self.register_buffer("sqrt_alphas_cumprod",            torch.sqrt(alphas_cumprod),          persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod",  torch.sqrt(1.0 - alphas_cumprod),    persistent=False)

    def _build_betas(self) -> torch.Tensor:
        if self.schedule_fn is not None:
            betas = self.schedule_fn(self.num_steps)
            if not torch.is_tensor(betas):
                raise TypeError(
                    f"schedule_fn must return a torch.Tensor, got {type(betas)}."
                )
            if betas.ndim != 1:
                raise ValueError(
                    f"schedule_fn must return a 1-D tensor, got shape {tuple(betas.shape)}."
                )
            if betas.numel() != self.num_steps:
                raise ValueError(
                    f"schedule_fn must return a tensor of length num_steps={self.num_steps}, "
                    f"got length {betas.numel()}."
                )
            if (betas <= 0).any() or (betas >= 1).any():
                raise ValueError(
                    "schedule_fn must return betas strictly in (0, 1)."
                )
            return betas.float()

        if self.schedule == "linear":
            return _linear_beta_schedule(
                num_steps=self.num_steps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
            )

        if self.schedule == "cosine":
            return _cosine_beta_schedule(
                num_steps=self.num_steps,
                s=self.cosine_s,
            )

        raise RuntimeError("Unreachable schedule branch.")

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Uniform discrete timestep sampling in [0, num_steps - 1].
        """
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    def _extract(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Gather timestep-specific coefficients and reshape them for broadcasting.
        """
        if t.dtype != torch.long:
            raise TypeError(
                f"GaussianDiffusionProcess expects integer timestep tensor, got dtype={t.dtype}."
            )
        if t.ndim != 1:
            raise ValueError(
                f"GaussianDiffusionProcess expects t with shape [B], got shape={tuple(t.shape)}."
            )
        if x_shape[0] != t.shape[0]:
            raise ValueError(
                "Batch size mismatch between x tensor and timestep tensor."
            )

        out = values.gather(0, t)
        reshape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
        return out.reshape(reshape)

    def forward_state(
        self,
        *,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Build:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a torch.Tensor.")
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor.")

        if noise is None:
            noise = torch.randn_like(x0)

        if noise.shape != x0.shape:
            raise ValueError(
                f"noise shape must match x0 shape. Got noise={tuple(noise.shape)}, x0={tuple(x0.shape)}."
            )

        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod,
            t,
            x0.shape,
        )
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x0.shape,
        )

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        # v = √ᾱ · ε - √(1-ᾱ) · x0
        # Used as supervision target when prediction_type="v".
        velocity_target = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * x0

        return {
            "x0": x0,
            "xt": xt,
            "t": t,
            "noise": noise,
            "cond": cond,
            "velocity_target": velocity_target,
            "process_aux": {
                "sqrt_alphas_cumprod_t": sqrt_alpha_bar,
                "sqrt_one_minus_alphas_cumprod_t": sqrt_one_minus_alpha_bar,
            },
        }

    def predict_x0(
        self,
        *,
        model_pred: Any,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Reconstruct x0 from model output according to prediction_type.
        """
        if not torch.is_tensor(model_pred):
            return None
        if not torch.is_tensor(xt):
            raise TypeError("xt must be a torch.Tensor.")
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor.")

        if self.prediction_type == "x0":
            x0_pred = model_pred

        elif self.prediction_type == "epsilon":
            sqrt_alpha_bar = self._extract(
                self.sqrt_alphas_cumprod,
                t,
                xt.shape,
            )
            sqrt_one_minus_alpha_bar = self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                xt.shape,
            )
            x0_pred = (xt - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar

        elif self.prediction_type == "v":
            # x0 = √ᾱ · x_t - √(1-ᾱ) · v_pred
            sqrt_alpha_bar = self._extract(
                self.sqrt_alphas_cumprod,
                t,
                xt.shape,
            )
            sqrt_one_minus_alpha_bar = self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                xt.shape,
            )
            x0_pred = sqrt_alpha_bar * xt - sqrt_one_minus_alpha_bar * model_pred

        else:
            raise RuntimeError("Unreachable prediction_type branch.")

        if self.clip_x0_pred:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        return x0_pred