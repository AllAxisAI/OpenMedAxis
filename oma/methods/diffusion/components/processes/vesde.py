from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseDiffusionProcess


class VESDEProcess(BaseDiffusionProcess, nn.Module):
    """
    Variance Exploding SDE process for score-based generative modeling.

    Forward process: q(x_t | x_0) = N(x_t; x_0, σ_t² I)

        x_t = x_0 + σ_t * ε,   ε ~ N(0, I)

    Unlike VP-SDE (GaussianDiffusionProcess), the mean of x_0 is preserved
    at every noise level — only the variance grows. This makes reconstruction
    straightforward: x_0 = x_t - σ_t * ε.

    Noise levels follow a geometric schedule:

        σ_t = σ_min * (σ_max / σ_min) ** (t / (T - 1)),   t = 0, 1, ..., T-1

    t=0 corresponds to the least noisy level (σ_min), t=T-1 to the most
    noisy (σ_max). Timesteps are discrete integers in [0, T-1].

    Training uses denoising score matching. With σ²-weighting (Song & Ermon,
    NCSNv2), predicting ε and predicting the normalized score are equivalent:

        score = ∇_x log q(x_t | x_0) = -(x_t - x_0) / σ_t² = -ε / σ_t

    The recommended objective is epsilon prediction (``prediction_type="epsilon"``):

        loss = ||ε_θ(x_t, t) - ε||²

    and the model is called at inference time as ``model(x_t, t, cond=cond)``.

    Inherits nn.Module so schedule buffers move automatically with the method.
    Buffers are non-persistent — recomputable from constructor arguments.

    Parameters
    ----------
    num_levels:
        Number of discrete noise levels T. Timesteps in [0, T-1].
    sigma_min:
        Lowest noise level σ_0. Typical: 0.01.
    sigma_max:
        Highest noise level σ_{T-1}. Typical: 50.0 (NCSNv2) or 100.0.
    prediction_type:
        What the model predicts. ``"epsilon"`` (recommended) or ``"score"``.

    References
    ----------
    Yang Song, Stefano Ermon,
    "Improved Techniques for Training Score-Based Generative Models",
    NeurIPS 2020. (NCSNv2)
    https://arxiv.org/abs/2006.09011

    Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar,
    Stefano Ermon, Ben Poole,
    "Score-Based Generative Modeling through Stochastic Differential Equations",
    ICLR 2021.
    https://arxiv.org/abs/2011.13456

    # TODO (advanced):
    # - Continuous-time VE-SDE with float t in [0, 1]
    # - VP-SDE process (continuous limit of DDPM)
    # - sub-VP SDE process
    # - Probability flow ODE (deterministic sampling, same marginals)
    # - Predictor-Corrector (PC) sampler (Song et al. 2021)
    # - Euler-Maruyama reverse SDE sampler
    """

    def __init__(
        self,
        num_levels: int = 232,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        prediction_type: str = "epsilon",
    ) -> None:
        super().__init__()

        if num_levels < 2:
            raise ValueError(f"num_levels must be >= 2, got {num_levels}.")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be > 0, got {sigma_min}.")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be > sigma_min, got sigma_max={sigma_max}, sigma_min={sigma_min}."
            )
        if prediction_type not in {"epsilon", "score"}:
            raise ValueError(
                f"Unsupported prediction_type '{prediction_type}'. "
                "Choose from ['epsilon', 'score']."
            )

        self.num_levels = int(num_levels)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.prediction_type = prediction_type

        # Geometric schedule: σ_t = σ_min * (σ_max / σ_min) ** (t / (T-1))
        sigmas = torch.exp(
            torch.linspace(
                start=torch.log(torch.tensor(sigma_min)),
                end=torch.log(torch.tensor(sigma_max)),
                steps=num_levels,
                dtype=torch.float32,
            )
        )

        self.register_buffer("sigmas", sigmas, persistent=False)

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
        """Uniform discrete timestep sampling in [0, num_levels - 1]."""
        return torch.randint(
            low=0,
            high=self.num_levels,
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
        Sample x_t from q(x_t | x_0):

            x_t = x_0 + σ_t * ε

        Parameters
        ----------
        x0:
            Clean sample, shape [B, ...].
        t:
            Discrete noise level indices in [0, num_levels - 1], shape [B].
        cond:
            Optional conditioning. Stored in state but not used by the process.
        noise:
            Optional pre-sampled noise. Sampled from N(0, I) if not provided.
        """
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a torch.Tensor.")
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor.")

        if noise is None:
            noise = torch.randn_like(x0)

        sigma_t = self._extract(t, x0.shape)
        xt = x0 + sigma_t * noise

        return {
            "x0":    x0,
            "xt":    xt,
            "t":     t,
            "noise": noise,
            "cond":  cond,
            "process_aux": {
                "sigma_t": sigma_t,
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
        Reconstruct x_0 from model output.

        prediction_type="epsilon":
            x_0 = x_t - σ_t * ε_θ

        prediction_type="score":
            x_0 = x_t + σ_t² * s_θ
            (since score = -(x_t - x_0) / σ_t²)
        """
        if not torch.is_tensor(model_pred):
            return None

        sigma_t = self._extract(t, xt.shape)

        if self.prediction_type == "epsilon":
            return xt - sigma_t * model_pred

        if self.prediction_type == "score":
            return xt + sigma_t ** 2 * model_pred

        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _extract(
        self,
        t: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """Gather per-sample σ_t and reshape for broadcasting."""
        sigma = self.sigmas[t]
        return sigma.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))
