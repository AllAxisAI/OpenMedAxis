from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


LossState = Dict[str, Any]
LogDict = Dict[str, torch.Tensor]


@dataclass
class LossOutput:
    """
    Output of a single loss term.

    Attributes
    ----------
    value:
        Weighted scalar loss tensor that will contribute to optimization.
    logs:
        Metrics/log values produced by this term.
    group:
        Optimization group this loss belongs to, e.g. 'main', 'disc'.
    raw_value:
        Unweighted scalar loss tensor before multiplying by term weight.
    name:
        Human-readable term name.
    """
    value: torch.Tensor
    logs: LogDict
    group: str = "main"
    raw_value: Optional[torch.Tensor] = None
    name: Optional[str] = None


class LossTerm(nn.Module):
    """
    Base class for a single atomic loss term.

    A loss term reads from a shared `state` dictionary and returns a `LossOutput`.

    Example state keys might be:
        - 'pred'
        - 'target'
        - 'recon'
        - 'input'
        - 'posterior'
        - 'latent'
        - 'global_step'
        - 'split'
        - 'last_layer'
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: Optional[str] = None,
        group: str = "main",
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.name = name or self.__class__.__name__.lower()
        self.group = group

    def compute(self, state: LossState) -> torch.Tensor:
        """
        Compute the raw unweighted scalar loss.

        Must return a scalar tensor.
        """
        raise NotImplementedError

    def build_logs(
        self,
        raw_value: torch.Tensor,
        weighted_value: torch.Tensor,
        state: LossState,
    ) -> LogDict:
        """
        Build per-term logs.

        By default logs both raw and weighted values.
        """
        split = state.get("split", "train")
        return {
            f"{split}/{self.name}": raw_value.detach(),
            f"{split}/{self.name}_weighted": weighted_value.detach(),
        }

    def validate(self, state: LossState) -> None:
        """
        Optional validation hook for checking required keys or shapes.
        """
        return None

    def forward(self, state: LossState) -> LossOutput:
        self.validate(state)

        raw_value = self.compute(state)
        if not torch.is_tensor(raw_value):
            raise TypeError(
                f"{self.__class__.__name__}.compute() must return a torch.Tensor, "
                f"got {type(raw_value)}"
            )

        if raw_value.ndim != 0:
            raw_value = raw_value.mean()

        weighted_value = self.weight * raw_value
        logs = self.build_logs(raw_value, weighted_value, state)

        return LossOutput(
            value=weighted_value,
            logs=logs,
            group=self.group,
            raw_value=raw_value,
            name=self.name,
        )


class StatefulLossTerm(LossTerm):
    """
    Optional extension point for terms that need to write useful intermediate
    values back into the shared state dict.

    Example use cases:
        - adversarial terms storing logits
        - perceptual terms caching converted inputs
        - specialized recipes exposing auxiliary diagnostics
    """

    def update_state(
        self,
        state: LossState,
        raw_value: torch.Tensor,
        weighted_value: torch.Tensor,
    ) -> None:
        return None

    def forward(self, state: LossState) -> LossOutput:
        out = super().forward(state)
        self.update_state(state, out.raw_value, out.value)
        return out