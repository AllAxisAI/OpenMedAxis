from __future__ import annotations

from typing import Optional

import torch

from ..base import LossState, LossTerm


class _BasePixelLossTerm(LossTerm):
    """
    Base class for pixel-domain losses operating on prediction and target tensors.
    """

    def __init__(
        self,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        name: Optional[str] = None,
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.pred_key = pred_key
        self.target_key = target_key

    def validate(self, state: LossState) -> None:
        if self.pred_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.pred_key}' in loss state."
            )
        if self.target_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.target_key}' in loss state."
            )

        pred = state[self.pred_key]
        target = state[self.target_key]

        if not torch.is_tensor(pred):
            raise TypeError(
                f"State key '{self.pred_key}' must be a torch.Tensor, got {type(pred)}"
            )
        if not torch.is_tensor(target):
            raise TypeError(
                f"State key '{self.target_key}' must be a torch.Tensor, got {type(target)}"
            )

        if pred.shape != target.shape:
            raise ValueError(
                f"{self.__class__.__name__} requires matching shapes, "
                f"got pred {tuple(pred.shape)} and target {tuple(target.shape)}"
            )

    def _get_tensors(self, state: LossState) -> tuple[torch.Tensor, torch.Tensor]:
        pred = state[self.pred_key]
        target = state[self.target_key]
        return pred, target


class L1LossTerm(_BasePixelLossTerm):
    """
    Mean absolute error loss.
    """

    def __init__(
        self,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        name: str = "l1",
        group: str = "main",
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name=name,
            group=group,
        )

    def compute(self, state: LossState) -> torch.Tensor:
        pred, target = self._get_tensors(state)
        return torch.abs(pred - target).mean()


class L2LossTerm(_BasePixelLossTerm):
    """
    Mean squared error loss.
    """

    def __init__(
        self,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        name: str = "l2",
        group: str = "main",
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name=name,
            group=group,
        )

    def compute(self, state: LossState) -> torch.Tensor:
        pred, target = self._get_tensors(state)
        return ((pred - target) ** 2).mean()


class CharbonnierLossTerm(_BasePixelLossTerm):
    """
    Differentiable robust L1-like loss:
        sqrt((x - y)^2 + eps^2)

    Useful when you want something smoother than L1 near zero.
    """

    def __init__(
        self,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        eps: float = 1e-6,
        name: str = "charbonnier",
        group: str = "main",
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name=name,
            group=group,
        )
        self.eps = float(eps)

    def compute(self, state: LossState) -> torch.Tensor:
        pred, target = self._get_tensors(state)
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class HuberLossTerm(_BasePixelLossTerm):
    """
    Smooth L1 / Huber loss.
    """

    def __init__(
        self,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        delta: float = 1.0,
        name: str = "huber",
        group: str = "main",
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            weight=weight,
            name=name,
            group=group,
        )
        self.delta = float(delta)

    def compute(self, state: LossState) -> torch.Tensor:
        pred, target = self._get_tensors(state)
        abs_diff = torch.abs(pred - target)

        quadratic = torch.minimum(
            abs_diff,
            torch.tensor(self.delta, device=abs_diff.device, dtype=abs_diff.dtype),
        )
        linear = abs_diff - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()