from __future__ import annotations

from typing import Optional

import torch

from ..base import LossState
from .pixel import _BasePixelLossTerm


class MinSnrWeightedLossTerm(_BasePixelLossTerm):
    """
    MSE loss with Min-SNR timestep weighting (Hang et al., 2023).

    Weights each batch element's loss by:

        w(t) = min(SNR(t), gamma) / SNR(t)

    where SNR(t) = alpha_bar_t / (1 - alpha_bar_t).

    At high t (near-pure noise): SNR is low  → w(t) ≈ 1   (full weight)
    At low  t (near-clean data): SNR is high → w(t) < 1   (downweighted)

    This prevents the network from over-focusing on easy low-noise timesteps
    and improves training efficiency.

    The weight tensor is a non-persistent buffer — it moves automatically
    when the parent method is moved to a device, and is not saved in
    checkpoints (it is recomputable from alpha_bar).

    Parameters
    ----------
    alpha_bar:
        Cumulative product of (1 - beta_t), shape ``[T]``.
        Typically ``process.alphas_cumprod`` from GaussianDiffusionProcess.
    pred_key:
        State key for model prediction.
    target_key:
        State key for supervision target.
    weight:
        Global scalar multiplier for this loss term.
    name:
        Log name. Default ``"min_snr_mse"``.
    group:
        Optimizer group. Default ``"main"``.
    gamma:
        SNR clipping threshold. Commonly 5.0 (paper default).
        Higher gamma = less downweighting of low-t steps.
    t_key:
        State key for the timestep tensor. Default ``"t"``.
        If not found in state, falls back to unweighted MSE.

    Example
    -------
    ::

        process = GaussianDiffusionProcess(num_steps=1000, schedule="cosine")
        objective = EpsilonObjective()

        loss_fn = LossComposer([
            MinSnrWeightedLossTerm(
                alpha_bar=process.alphas_cumprod,
                pred_key=objective.pred_key,
                target_key=objective.target_key,
                gamma=5.0,
            )
        ])

    Or via the objective helper:

    ::

        loss_fn = LossComposer([
            objective.default_loss_term()
                .with_snr_weighting(process.alphas_cumprod)  # future convenience
        ])
    """

    def __init__(
        self,
        alpha_bar: torch.Tensor,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        name: str = "min_snr_mse",
        group: str = "main",
        gamma: float = 5.0,
        t_key: str = "t",
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name=name,
            group=group,
        )

        if not torch.is_tensor(alpha_bar):
            raise TypeError(
                f"alpha_bar must be a torch.Tensor, got {type(alpha_bar)}."
            )
        if alpha_bar.ndim != 1:
            raise ValueError(
                f"alpha_bar must be 1-D, got shape {tuple(alpha_bar.shape)}."
            )
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}.")

        self.gamma = float(gamma)
        self.t_key = t_key

        ab = alpha_bar.float()
        snr = ab / (1.0 - ab).clamp(min=1e-8)
        loss_weights = torch.minimum(snr, snr.new_full(snr.shape, gamma)) / snr

        self.register_buffer("loss_weights", loss_weights, persistent=False)

    def compute(self, state: LossState) -> torch.Tensor:
        pred, target = self._get_tensors(state)
        t = state.get(self.t_key, None)

        mse = (pred - target) ** 2

        if t is None or not torch.is_tensor(t):
            return mse.mean()

        w = self.loss_weights[t]
        # reshape [B] → [B, 1, 1, ...] to broadcast over spatial + channel dims
        for _ in range(pred.ndim - 1):
            w = w.unsqueeze(-1)

        return (w * mse).mean()
