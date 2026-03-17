from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..base import LossState, LossTerm


class LPIPSLossTerm(LossTerm):
    """
    LPIPS perceptual loss term.

    Notes
    -----
    - LPIPS usually expects 3-channel inputs in roughly [-1, 1].
    - For grayscale medical images, `repeat_grayscale=True` repeats the single
      channel to 3 channels.
    - The provided lpips_model is expected to return either:
        * a tensor of shape [B, 1, H, W] / [B, 1] / scalar
        * or another tensor reducible by `.mean()`
    """

    def __init__(
        self,
        lpips_model: nn.Module,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        name: str = "lpips",
        group: str = "main",
        repeat_grayscale: bool = True,
        normalize_inputs: bool = False,
        clamp_range: Optional[tuple[float, float]] = None,
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.lpips_model = lpips_model
        self.pred_key = pred_key
        self.target_key = target_key
        self.repeat_grayscale = repeat_grayscale
        self.normalize_inputs = normalize_inputs
        self.clamp_range = clamp_range

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

        if pred.ndim != 4:
            raise ValueError(
                f"{self.__class__.__name__} expects 4D tensors [B, C, H, W], "
                f"got shape {tuple(pred.shape)}"
            )

    def _maybe_clamp(self, x: torch.Tensor) -> torch.Tensor:
        if self.clamp_range is None:
            return x
        low, high = self.clamp_range
        return x.clamp(min=low, max=high)

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return x

        # Per-sample min-max normalize to [-1, 1]
        b = x.shape[0]
        x_flat = x.view(b, -1)
        x_min = x_flat.min(dim=1)[0].view(b, 1, 1, 1)
        x_max = x_flat.max(dim=1)[0].view(b, 1, 1, 1)

        denom = (x_max - x_min).clamp_min(1e-8)
        x = (x - x_min) / denom
        x = 2.0 * x - 1.0
        return x

    def _maybe_repeat_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1 and self.repeat_grayscale:
            x = x.repeat(1, 3, 1, 1)
        return x

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_clamp(x)
        x = self._maybe_normalize(x)
        x = self._maybe_repeat_channels(x)
        return x.contiguous()

    def compute(self, state: LossState) -> torch.Tensor:
        pred = state[self.pred_key]
        target = state[self.target_key]

        pred = self._prepare(pred)
        target = self._prepare(target)

        value = self.lpips_model(pred, target)

        if not torch.is_tensor(value):
            raise TypeError(
                f"{self.__class__.__name__}: lpips_model must return a torch.Tensor, "
                f"got {type(value)}"
            )

        return value.mean()


class FeatureExtractorPerceptualLossTerm(LossTerm):
    """
    Generic feature-space perceptual loss using a feature extractor.

    The feature extractor may return:
    - a single tensor
    - a list/tuple of tensors
    - a dict of tensors

    This is useful when you later want medical-domain perceptual losses
    instead of LPIPS.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        pred_key: str = "pred",
        target_key: str = "target",
        weight: float = 1.0,
        name: str = "feature_perceptual",
        group: str = "main",
        criterion: str = "l1",
        detach_target: bool = True,
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        if criterion not in ("l1", "l2"):
            raise ValueError(f"Unsupported criterion '{criterion}'. Use 'l1' or 'l2'.")
        self.feature_extractor = feature_extractor
        self.pred_key = pred_key
        self.target_key = target_key
        self.criterion = criterion
        self.detach_target = detach_target

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

    def _reduce_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.detach_target:
            b = b.detach()

        if self.criterion == "l1":
            return torch.abs(a - b).mean()
        return ((a - b) ** 2).mean()

    def _to_sequence(self, features):
        if torch.is_tensor(features):
            return [features]
        if isinstance(features, (list, tuple)):
            return list(features)
        if isinstance(features, dict):
            return [features[k] for k in sorted(features.keys())]
        raise TypeError(
            f"{self.__class__.__name__}: feature extractor returned unsupported type {type(features)}"
        )

    def compute(self, state: LossState) -> torch.Tensor:
        pred = state[self.pred_key]
        target = state[self.target_key]

        pred_feats = self._to_sequence(self.feature_extractor(pred))
        target_feats = self._to_sequence(self.feature_extractor(target))

        if len(pred_feats) != len(target_feats):
            raise ValueError(
                f"{self.__class__.__name__}: feature extractor returned different numbers "
                f"of feature maps for pred ({len(pred_feats)}) and target ({len(target_feats)})."
            )

        loss = pred.new_zeros(())
        for pf, tf in zip(pred_feats, target_feats):
            if not torch.is_tensor(pf) or not torch.is_tensor(tf):
                raise TypeError(
                    f"{self.__class__.__name__}: feature maps must be tensors."
                )
            if pf.shape != tf.shape:
                raise ValueError(
                    f"{self.__class__.__name__}: feature shapes must match, "
                    f"got {tuple(pf.shape)} vs {tuple(tf.shape)}"
                )
            loss = loss + self._reduce_pair(pf, tf)

        return loss