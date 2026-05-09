from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDiffusionProcess


def _linear_betas(
    num_steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def _cosine_betas(
    num_steps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-8, max=max_beta)


class I2SBProcess(BaseDiffusionProcess, nn.Module):
    """
    Discrete-time Image-to-Image Schrödinger Bridge process.

    Forward marginal connecting two image distributions x_0 (target) and
    x_1 (source / corrupted image):

        q(x_t | x_0, x_1) = N(x_t; m_t, C_t * I)

    where the coefficients are derived from the cumulative beta schedule:

        c_t    = Σ_{i=1}^{t} β_i          (cumulative sum up to t)
        c̄_t   = c_T - c_t                  (remaining cumulative sum)

        m_t    = (c̄_t / c_T) * x_0 + (c_t / c_T) * x_1
        C_t    = c_t * c̄_t / c_T           (bridge variance at t)

    At t=0: m_0 = x_0, C_0 = 0  (pure target).
    At t=T: m_T = x_1, C_T = 0  (pure source).
    Noise is maximal at the midpoint.

    The reverse posterior follows from standard Gaussian bridge conditioning:

        q(x_{t-1} | x_t, x_0, x_1) = N(x_{t-1}; μ_post, σ²_post * I)

        μ_post    = m_{t-1} + (C_{t-1} / C_t) * (x_t - m_t)
        σ²_post   = C_{t-1} * (C_t - C_{t-1}) / C_t

    Unlike SelfRDB, this process works with any standard DDPM-style beta
    schedule and uses cleaner linear-algebraic bridge coefficients.
    The model predicts x_0 given (x_t, x_1, t); how x_1 is passed to the
    model (concatenation, cross-attention, etc.) is left to the Method.

    Inherits nn.Module so schedule buffers move automatically to the correct
    device when the parent method is moved (e.g. method.to("cuda")).
    Buffers are non-persistent — they are recomputable from constructor args.

    Parameters
    ----------
    num_steps:
        Number of diffusion timesteps. Timesteps indexed in [1, num_steps].
    schedule:
        Beta schedule type: ``"linear"`` or ``"cosine"``. Ignored when
        ``schedule_fn`` is provided.
    beta_start:
        Linear schedule start value. Default 1e-4.
    beta_end:
        Linear schedule end value. Default 2e-2.
    cosine_s:
        Offset for cosine schedule. Default 0.008.
    schedule_fn:
        Optional custom callable ``(num_steps: int) -> Tensor`` returning
        betas of length ``num_steps`` with values in (0, 1).

    References
    ----------
    Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A. Theodorou,
    Weili Nie, "I2SB: Image-to-Image Schrödinger Bridge", ICML 2023.
    https://arxiv.org/abs/2302.05872
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
        schedule_fn: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}.")

        self.num_steps = int(num_steps)
        self.schedule = schedule.lower()
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.cosine_s = float(cosine_s)
        self.schedule_fn = schedule_fn

        if schedule_fn is None and self.schedule not in {"linear", "cosine"}:
            raise ValueError(
                f"Unsupported schedule '{schedule}'. "
                "Choose from ['linear', 'cosine'] or pass a custom schedule_fn."
            )

        betas = self._build_betas()

        # Cumulative sums: c_t[i] = Σ β_j for j=1..i
        # Prepend 0 so index t directly gives the cumsum up to step t.
        c = torch.cat([torch.zeros(1, dtype=torch.float32), torch.cumsum(betas, dim=0)])
        c_T = c[-1]
        c_bar = c_T - c  # remaining cumsum from t to T

        # Forward marginal coefficients
        # m_t = (c_bar_t / c_T) * x0 + (c_t / c_T) * x1
        coef_x0 = c_bar / c_T       # shape [T+1]
        coef_x1 = c / c_T           # shape [T+1]

        # Bridge variance: C_t = c_t * c_bar_t / c_T
        # std_t = sqrt(C_t), clamped for numerical safety
        bridge_var = c * c_bar / c_T
        bridge_std = bridge_var.clamp(min=0.0).sqrt()

        self.register_buffer("c",           c,           persistent=False)
        self.register_buffer("c_bar",       c_bar,       persistent=False)
        self.register_buffer("bridge_var",  bridge_var,  persistent=False)
        self.register_buffer("bridge_std",  bridge_std,  persistent=False)
        self.register_buffer("coef_x0",     coef_x0,     persistent=False)
        self.register_buffer("coef_x1",     coef_x1,     persistent=False)

    # ------------------------------------------------------------------
    # schedule construction
    # ------------------------------------------------------------------
    def _build_betas(self) -> torch.Tensor:
        if self.schedule_fn is not None:
            betas = self.schedule_fn(self.num_steps)
            if not torch.is_tensor(betas):
                raise TypeError(
                    f"schedule_fn must return a torch.Tensor, got {type(betas)}."
                )
            if betas.ndim != 1 or betas.numel() != self.num_steps:
                raise ValueError(
                    f"schedule_fn must return a 1-D tensor of length {self.num_steps}."
                )
            return betas.float()

        if self.schedule == "linear":
            return _linear_betas(self.num_steps, self.beta_start, self.beta_end)

        return _cosine_betas(self.num_steps, self.cosine_s)

    # ------------------------------------------------------------------
    # BaseDiffusionProcess interface
    # ------------------------------------------------------------------
    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Uniform discrete timestep sampling in [1, num_steps]."""
        return torch.randint(
            low=1,
            high=self.num_steps + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

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
        Sample x_t from q(x_t | x_0, x_1):

            x_t = coef_x0(t) * x0 + coef_x1(t) * x1 + bridge_std(t) * eps

        Parameters
        ----------
        x0:
            Target image (clean), shape [B, C, H, W].
        t:
            Integer timesteps in [1, num_steps], shape [B].
        cond:
            Source image x_1 (corrupted / paired input), same shape as x0.
        noise:
            Optional pre-sampled noise. Sampled from N(0, I) if not provided.
        """
        x1 = cond
        if x1 is None:
            raise ValueError(
                "I2SBProcess.forward_state requires cond (source image x_1)."
            )
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a torch.Tensor.")
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor.")
        if not torch.is_tensor(x1):
            raise TypeError("cond (x1) must be a torch.Tensor.")

        if noise is None:
            noise = torch.randn_like(x0)

        shape = [-1] + [1] * (x0.ndim - 1)
        coef_x0_t   = self.coef_x0[t].view(shape)
        coef_x1_t   = self.coef_x1[t].view(shape)
        bridge_std_t = self.bridge_std[t].view(shape)

        xt = coef_x0_t * x0 + coef_x1_t * x1 + bridge_std_t * noise

        return {
            "x0":    x0,
            "xt":    xt.detach(),
            "t":     t,
            "noise": noise,
            "cond":  x1,
            "process_aux": {
                "coef_x0_t":    coef_x0_t,
                "coef_x1_t":    coef_x1_t,
                "bridge_std_t": bridge_std_t,
            },
        }

    # ------------------------------------------------------------------
    # reverse / posterior step
    # ------------------------------------------------------------------
    def q_posterior(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample q(x_{t-1} | x_t, x_0, x_1) — one reverse step.

        Posterior mean:
            μ_post = m_{t-1} + (C_{t-1} / C_t) * (x_t - m_t)

        Posterior variance:
            σ²_post = C_{t-1} * (C_t - C_{t-1}) / C_t
        """
        shape = [-1] + [1] * (x0.ndim - 1)

        coef_x0_t    = self.coef_x0[t].view(shape)
        coef_x1_t    = self.coef_x1[t].view(shape)
        coef_x0_tm1  = self.coef_x0[t - 1].view(shape)
        coef_x1_tm1  = self.coef_x1[t - 1].view(shape)

        C_t   = self.bridge_var[t].view(shape)
        C_tm1 = self.bridge_var[t - 1].view(shape)

        m_t   = coef_x0_t  * x0 + coef_x1_t  * x1
        m_tm1 = coef_x0_tm1 * x0 + coef_x1_tm1 * x1

        # Posterior mean: prior mean at t-1 corrected by deviation at t
        ratio = C_tm1 / C_t.clamp(min=1e-8)
        mu_post = m_tm1 + ratio * (x_t - m_t)

        # Posterior variance
        var_post = (C_tm1 * (C_t - C_tm1) / C_t.clamp(min=1e-8)).clamp(min=0.0)

        return mu_post + var_post.sqrt() * torch.randn_like(x_t)

    # ------------------------------------------------------------------
    # full reverse sampling
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample_x0(
        self,
        x1: torch.Tensor,
        model: Any,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full reverse sampling: draw x_0 ~ p(x_0 | x_1).

        Parameters
        ----------
        x1:
            Source image (the corrupted / paired input), shape [B, C, H, W].
        model:
            x0-prediction network. Called as ``model(x_t, t, cond=x1)``
            by default. Override in a subclass or Method if a different
            calling convention is needed (e.g. concatenation).
        cond:
            Additional conditioning passed to the model alongside x1.
            Unused in the default implementation; available for subclasses.
        """
        B = x1.shape[0]
        device = x1.device

        # Start at x_T = x_1 (deterministic, bridge variance at T is 0)
        x_t = x1.clone()

        timesteps = torch.arange(self.num_steps, 0, -1, device=device)

        for t_scalar in timesteps:
            t = t_scalar.expand(B)

            # Predict x0
            x0_pred = model(x_t, t, cond=x1)

            if t_scalar.item() == 1:
                x_t = x0_pred
            else:
                x_t = self.q_posterior(t, x_t, x0_pred, x1)

        return x_t
