from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseTimeSampler


class ImportanceTimeSampler(BaseTimeSampler, nn.Module):
    """
    Importance-weighted discrete timestep sampler.

    Samples timesteps proportional to a provided weight vector, allowing
    the training distribution over t to be shaped by any criterion.

    Common use cases
    ----------------
    - **Min-SNR weighting** (Hang et al., 2023): downweight high-t steps
      where the SNR is very low and the loss signal is noisy.
    - **Loss-proportional sampling**: sample more from timesteps where
      the current model loss is highest.
    - **SNR-uniform sampling**: reweight so that each SNR level is seen
      equally often, regardless of the schedule shape.

    The weights are registered as a non-persistent buffer so they move
    automatically when the parent method is moved to a device.

    Parameters
    ----------
    weights:
        1-D tensor of shape ``[num_steps]`` with non-negative values.
        Does not need to be normalised — softmax is applied internally.

    Example: Min-SNR sampler
    ------------------------
    ::

        process = GaussianDiffusionProcess(num_steps=1000, schedule="cosine")
        snr = process.alphas_cumprod / (1 - process.alphas_cumprod)
        weights = torch.minimum(snr, torch.full_like(snr, 5.0))

        method = GaussianDiffusionMethod(
            process=process,
            time_sampler=ImportanceTimeSampler(weights),
            ...
        )
    """

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()

        if not torch.is_tensor(weights):
            raise TypeError("weights must be a torch.Tensor.")
        if weights.ndim != 1:
            raise ValueError(
                f"weights must be 1-D, got shape {tuple(weights.shape)}."
            )
        if weights.numel() == 0:
            raise ValueError("weights must not be empty.")
        if (weights < 0).any():
            raise ValueError("weights must be non-negative.")

        probs = weights.float() / weights.float().sum()
        self.register_buffer("probs", probs, persistent=False)

    @property
    def num_steps(self) -> int:
        return int(self.probs.numel())

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return torch.multinomial(
            self.probs.to(device),
            num_samples=batch_size,
            replacement=True,
        )
