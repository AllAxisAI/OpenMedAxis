from __future__ import annotations

from typing import Any, Optional

import torch

from .base import BaseDiffusionSampler


class AnnealedLangevinSampler(BaseDiffusionSampler):
    """
    Annealed Langevin dynamics sampler for VE-SDE / score-based models.

    Iterates from the highest noise level (σ_max) down to the lowest (σ_min).
    At each noise level it runs ``n_steps`` Langevin MCMC steps:

        z   ~ N(0, I)
        α_t  = step_size * (σ_t / σ_min)²
        x   ← x + α_t * s_θ(x, t) + √(2 α_t) * z

    where s_θ is the score function. If the model predicts ε (epsilon) rather
    than the raw score, the score is recovered as:

        s_θ = -ε_θ / σ_t

    After the final noise level, an optional deterministic denoising step
    (``denoise_last=True``) replaces the stochastic Langevin step with a
    direct x_0 reconstruction — standard practice that removes the last
    residual noise artifact.

    Parameters
    ----------
    n_steps:
        Number of Langevin steps per noise level. Higher → better samples,
        slower inference. NCSNv2 paper uses 100. Default 10 for fast runs.
    step_size:
        Base step size ε. Scales with (σ_t / σ_min)² at each level.
        NCSNv2 tunes this per dataset; start with 2e-5 and sweep.
    denoise_last:
        If True, the last step is a deterministic x_0 prediction instead of
        a stochastic Langevin step. Recommended (True by default).
    noise_level_order:
        ``"descending"`` (default): start from σ_max, move to σ_min —
        standard generative sampling from noise.
        ``"ascending"``: start from σ_min — useful for debugging or
        refinement tasks.

    Notes
    -----
    The process must be a ``VESDEProcess`` (or compatible): it must expose
    a ``sigmas`` buffer and a ``predict_x0`` method.

    References
    ----------
    Yang Song, Stefano Ermon,
    "Improved Techniques for Training Score-Based Generative Models",
    NeurIPS 2020. (NCSNv2, Algorithm 1)
    https://arxiv.org/abs/2006.09011

    # TODO (advanced):
    # - Predictor-Corrector (PC) sampler: interleave reverse SDE steps
    #   (predictor) with corrector Langevin steps at each noise level.
    #   (Song et al., ICLR 2021, Algorithm 1/2)
    # - Euler-Maruyama reverse SDE sampler (single stochastic step / level)
    # - Probability flow ODE sampler (deterministic, same marginals as SDE)
    """

    def __init__(
        self,
        n_steps: int = 10,
        step_size: float = 2e-5,
        denoise_last: bool = True,
        noise_level_order: str = "descending",
    ) -> None:
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}.")
        if noise_level_order not in {"descending", "ascending"}:
            raise ValueError(
                f"noise_level_order must be 'descending' or 'ascending', "
                f"got '{noise_level_order}'."
            )

        self.n_steps = int(n_steps)
        self.step_size = float(step_size)
        self.denoise_last = bool(denoise_last)
        self.noise_level_order = noise_level_order

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
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Run annealed Langevin dynamics.

        Parameters
        ----------
        model:
            Score / epsilon-prediction network.
        process:
            A ``VESDEProcess`` instance — must expose ``sigmas`` and
            ``predict_x0`` (used for the optional last denoising step).
        cond:
            Optional conditioning.
        shape:
            Sample shape. Required unless ``x_init`` is provided.
        x_init:
            Optional starting point. If not provided, x is initialized to
            Gaussian noise scaled by σ_max.
        method:
            Optional owning method (for routing through forward_model).
        """
        if not hasattr(process, "sigmas"):
            raise AttributeError(
                "AnnealedLangevinSampler requires a VESDEProcess with a 'sigmas' buffer."
            )

        device, dtype = self.get_model_device_and_dtype(model)

        # Initial state: x_T ~ N(0, σ_max² I)
        if x_init is not None:
            x = x_init.to(device=device, dtype=dtype)
        else:
            if shape is None:
                raise ValueError(
                    "AnnealedLangevinSampler requires either x_init or shape."
                )
            x = torch.randn(shape, device=device, dtype=dtype) * process.sigmas[-1]

        batch_size = x.shape[0]
        T = process.sigmas.shape[0]
        sigma_min = process.sigmas[0]

        # Build time index sequence
        indices = torch.arange(T, device=device)
        if self.noise_level_order == "descending":
            indices = indices.flip(0)  # T-1 down to 0

        for idx in indices:
            t = self.make_time_tensor(
                t_value=int(idx.item()),
                batch_size=batch_size,
                device=device,
                dtype=torch.long,
            )

            sigma_t = process.sigmas[idx]
            alpha_t = self.step_size * (sigma_t / sigma_min) ** 2

            is_last = (idx == indices[-1])

            if is_last and self.denoise_last:
                # Deterministic final denoising step (no noise)
                model_pred = self.call_model(
                    model=model, x=x, t=t, cond=cond, method=method, **kwargs
                )
                x0 = self.reconstruct_clean(
                    model_pred=model_pred,
                    x=x,
                    t=t,
                    cond=cond,
                    process=process,
                    method=method,
                )
                if x0 is not None:
                    x = x0
                continue

            for _ in range(self.n_steps):
                model_pred = self.call_model(
                    model=model, x=x, t=t, cond=cond, method=method, **kwargs
                )

                # Convert to score if model predicts epsilon
                score = self._to_score(model_pred, sigma_t, process)

                z = torch.randn_like(x)
                x = x + alpha_t * score + (2.0 * alpha_t).sqrt() * z

        return x

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _to_score(
        self,
        model_pred: torch.Tensor,
        sigma_t: torch.Tensor,
        process: Any,
    ) -> torch.Tensor:
        """
        Convert model output to score function s_θ = ∇_x log p(x_t).

        prediction_type="epsilon":  score = -ε_θ / σ_t
        prediction_type="score":    score = s_θ  (returned as-is)
        """
        pred_type = getattr(process, "prediction_type", "epsilon")

        if pred_type == "epsilon":
            return -model_pred / sigma_t

        # "score": model already outputs the score
        return model_pred
