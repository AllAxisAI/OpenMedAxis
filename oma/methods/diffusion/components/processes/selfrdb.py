from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseDiffusionProcess


class SelfRDBProcess(BaseDiffusionProcess, nn.Module):
    """
    Diffusion bridge process from the SelfRDB paper.

    Implements the bridge q(x_t | x_0, y) where both endpoints are known:
        - x_0: target image
        - y:   source / conditioning image

    The forward process interpolates between the two with a symmetric noise
    schedule (narrow at both ends, peaked in the middle):

        x_t = mu_x0(t) * x0 + mu_y(t) * y + std(t) * eps,  eps ~ N(0, I)

    The schedule coefficients (mu_x0, mu_y, std) are derived from the
    Gaussian product of two cumulative beta sequences — see _build_betas and
    _gaussian_product for details.

    Inherits nn.Module so schedule buffers move automatically to the correct
    device when the parent method is moved (e.g. method.to("cuda")).
    Buffers are non-persistent — they are recomputable from constructor args.

    Parameters
    ----------
    n_steps:
        Number of diffusion timesteps. Timesteps are indexed in [1, n_steps].
    beta_start:
        Start value for the beta schedule (before squaring).
    beta_end:
        End value for the beta schedule (before scaling by n_steps and squaring).
    gamma:
        Controls the noise magnitude at the bridge midpoint. Higher = noisier.
    n_recursions:
        Default number of recursive self-consistency refinement steps.
        Set to 1 for standard single-pass x0 prediction.
        Can be overridden per-call in sample_x0.
    consistency_threshold:
        Early stopping threshold for recursive refinement. Stops when the
        max mean L1 change between consecutive x0 estimates falls below this.
        Set to 0.0 to always run all n_recursions steps.

    Notes
    -----
    The schedule is symmetric: betas are built so that the bridge is narrow
    at t=0 (pure x0) and t=T (pure y), and widest in the middle.

    References
    ----------
    Fuat Arslan, Bilal Kabas, Onat Dalmaz, Muzaffer Ozbey, Tolga Çukur,
    "Self-consistent recursive diffusion bridge for medical image translation",
    Medical Image Analysis, Volume 106, 2025, 103747, ISSN 1361-8415,
    https://doi.org/10.1016/j.media.2025.103747
    """

    def __init__(
        self,
        n_steps: int = 10,
        beta_start: float = 0.1,
        beta_end: float = 3.0,
        gamma: float = 1.0,
        n_recursions: int = 1,
        consistency_threshold: float = 0.01,
    ) -> None:
        super().__init__()

        if n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {n_steps}.")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}.")
        if n_recursions <= 0:
            raise ValueError(f"n_recursions must be > 0, got {n_recursions}.")

        self.n_steps = int(n_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.gamma_param = float(gamma)
        self.n_recursions = int(n_recursions)
        self.consistency_threshold = float(consistency_threshold)

        betas = self._build_betas()

        s = np.cumsum(betas) ** 0.5
        s_bar = np.flip(np.cumsum(betas)) ** 0.5
        mu_x0, mu_y, _ = self._gaussian_product(s, s_bar)

        gamma_scaled = gamma * float(betas.sum())
        std = gamma_scaled * s / (s ** 2 + s_bar ** 2)

        self.register_buffer("s",     torch.tensor(s.copy(),     dtype=torch.float32), persistent=False)
        self.register_buffer("mu_x0", torch.tensor(mu_x0.copy(), dtype=torch.float32), persistent=False)
        self.register_buffer("mu_y",  torch.tensor(mu_y.copy(),  dtype=torch.float32), persistent=False)
        self.register_buffer("std",   torch.tensor(std.copy(),   dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------
    # schedule construction
    # ------------------------------------------------------------------
    def _build_betas(self) -> np.ndarray:
        beta_end_scaled = self.beta_end / self.n_steps
        betas_len = self.n_steps + 1

        betas = np.linspace(
            self.beta_start ** 0.5,
            beta_end_scaled ** 0.5,
            betas_len,
        ) ** 2

        betas = np.append(0.0, betas).astype(np.float32)

        if betas_len % 2 == 1:
            betas = np.concatenate([
                betas[:betas_len // 2],
                [betas[betas_len // 2]],
                np.flip(betas[:betas_len // 2]),
            ])
        else:
            betas = np.concatenate([
                betas[:betas_len // 2],
                np.flip(betas[:betas_len // 2]),
            ])

        return betas

    @staticmethod
    def _gaussian_product(
        sigma1: np.ndarray,
        sigma2: np.ndarray,
    ):
        denom = sigma1 ** 2 + sigma2 ** 2
        mu1 = sigma2 ** 2 / denom
        mu2 = sigma1 ** 2 / denom
        var = (sigma1 ** 2 * sigma2 ** 2) / denom
        return mu1, mu2, var

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
        """Uniform discrete timestep sampling in [1, n_steps]."""
        return torch.randint(
            low=1,
            high=self.n_steps + 1,
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
        Sample x_t from q(x_t | x_0, source):

            x_t = mu_x0(t) * x0 + mu_y(t) * source + std(t) * eps

        Parameters
        ----------
        x0:
            Target image (clean sample), shape [B, C, H, W].
        t:
            Integer timesteps in [1, n_steps], shape [B].
        cond:
            Source / conditioning image, same shape as x0.
        noise:
            Optional pre-sampled noise. Sampled from N(0, I) if not provided.
        """
        source = cond
        if source is None:
            raise ValueError(
                "SelfRDBProcess.forward_state requires cond (source image)."
            )
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a torch.Tensor.")
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor.")
        if not torch.is_tensor(source):
            raise TypeError("cond (source) must be a torch.Tensor.")

        if noise is None:
            noise = torch.randn_like(x0)

        shape = [-1] + [1] * (x0.ndim - 1)
        mu_x0_t = self.mu_x0[t].view(shape)
        mu_y_t  = self.mu_y[t].view(shape)
        std_t   = self.std[t].view(shape)

        xt = mu_x0_t * x0 + mu_y_t * source + std_t * noise

        return {
            "x0":   x0,
            "xt":   xt.detach(),
            "t":    t,
            "noise": noise,
            "cond": source,
            "process_aux": {
                "mu_x0_t": mu_x0_t,
                "mu_y_t":  mu_y_t,
                "std_t":   std_t,
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
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample p(x_{t-1} | x_t, x0, source).

        Used during iterative reverse sampling.
        """
        shape = [-1] + [1] * (x0.ndim - 1)

        std_t     = self.s[t].view(shape)
        std_tm1   = self.s[t - 1].view(shape)
        mu_x0_t   = self.mu_x0[t].view(shape)
        mu_x0_tm1 = self.mu_x0[t - 1].view(shape)
        mu_y_t    = self.mu_y[t].view(shape)
        mu_y_tm1  = self.mu_y[t - 1].view(shape)

        var_t     = std_t ** 2
        var_tm1   = std_tm1 ** 2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1) ** 2
        v         = var_t_tm1 * (var_tm1 / var_t)

        mean = (
            mu_x0_tm1 * x0
            + mu_y_tm1 * source
            + ((var_tm1 - v) / var_t).sqrt()
            * (x_t - mu_x0_t * x0 - mu_y_t * source)
        )

        return mean + v.sqrt() * torch.randn_like(x_t)

    # ------------------------------------------------------------------
    # full reverse sampling
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample_x0(
        self,
        source: torch.Tensor,
        model: Any,
        n_recursions: Optional[int] = None,
        consistency_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Full reverse sampling: draw x_0 ~ p(x_0 | source).

        Parameters
        ----------
        source:
            Conditioning image, shape [B, C, H, W].
        model:
            Network with signature: model(cat(x_t, source), t, x_r=x0_r).
        n_recursions:
            Recursive refinement steps per timestep. Defaults to self.n_recursions.
        consistency_threshold:
            Early stopping threshold. Defaults to self.consistency_threshold.
        """
        if n_recursions is None:
            n_recursions = self.n_recursions
        if consistency_threshold is None:
            consistency_threshold = self.consistency_threshold

        timesteps = torch.arange(self.n_steps, 0, -1, device=source.device)
        timesteps = timesteps.unsqueeze(1).repeat(1, source.shape[0])

        # x_T: start from q(x_T | x0=0, source)
        t_init = timesteps[0]
        x_t = self.forward_state(
            x0=torch.zeros_like(source),
            t=t_init,
            cond=source,
        )["xt"]

        x0_pred = torch.zeros_like(source)
        for t in timesteps:
            x0_r = torch.zeros_like(x_t)
            for _ in range(n_recursions):
                x0_rp1 = model(torch.cat((x_t, source), dim=1), t, x_r=x0_r)

                if consistency_threshold > 0.0:
                    change = torch.abs(x0_rp1 - x0_r).mean(dim=0).max()
                    x0_r = x0_rp1
                    if change.item() < consistency_threshold:
                        break
                else:
                    x0_r = x0_rp1

            x0_pred = x0_r
            x_t = self.q_posterior(t, x_t, x0_pred, source)

        return x0_pred
