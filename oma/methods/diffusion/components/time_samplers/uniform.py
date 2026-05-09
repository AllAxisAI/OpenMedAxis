from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import BaseTimeSampler


class UniformTimeSampler(BaseTimeSampler):
    """
    Uniform discrete timestep sampler.

    Samples t ~ Uniform{low, ..., num_steps - 1} independently for each
    item in the batch. This matches the default behavior of
    GaussianDiffusionProcess.sample_time() and is the standard DDPM
    training schedule.

    Use this when you want time sampling to be explicit and configurable
    without changing the process.

    Parameters
    ----------
    num_steps:
        Upper bound (exclusive) of the timestep range.
    low:
        Lower bound (inclusive). Default 0.

    Example
    -------
    ::

        method = GaussianDiffusionMethod(
            process=GaussianDiffusionProcess(num_steps=1000),
            time_sampler=UniformTimeSampler(num_steps=1000),
            ...
        )
    """

    def __init__(self, num_steps: int, low: int = 0) -> None:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}.")
        if low < 0:
            raise ValueError(f"low must be >= 0, got {low}.")
        if low >= num_steps:
            raise ValueError(
                f"low must be < num_steps, got low={low}, num_steps={num_steps}."
            )
        self.num_steps = int(num_steps)
        self.low = int(low)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return torch.randint(
            low=self.low,
            high=self.num_steps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
