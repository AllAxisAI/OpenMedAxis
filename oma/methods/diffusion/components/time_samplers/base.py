from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseTimeSampler(ABC):
    """
    Base contract for time/timestep samplers in OpenMedAxis.

    A time sampler owns the distribution from which training timesteps
    are drawn. It is intentionally decoupled from the process so that
    timestep sampling can be changed without touching the forward path.

    Common research use cases
    -------------------------
    - Uniform sampling           : UniformTimeSampler
    - Importance / loss-weighted : ImportanceTimeSampler
    - Low-discrepancy / quasi-random: subclass this
    - Curriculum (easy → hard)   : subclass this

    Notes
    -----
    - Concrete implementations with tensor state (e.g. weight buffers)
      should also inherit nn.Module and use register_buffer so that
      method.to(device) moves them automatically.
    - The `state` argument is optional context from the current training
      step. Most samplers will ignore it; adaptive samplers may use it.
    """

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Sample a batch of timesteps.

        Parameters
        ----------
        batch_size:
            Number of timesteps to sample (one per item in the batch).
        device:
            Target device for the returned tensor.
        stage:
            Current training stage: ``"train"``, ``"val"``, or ``"test"``.
        state:
            Optional shared state dict from the current step. Adaptive
            samplers may read it; most samplers ignore it.

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size]``. Dtype is sampler-defined (typically
            ``torch.long`` for discrete, ``torch.float32`` for continuous).
        """
        raise NotImplementedError
