from __future__ import annotations

import torch
import torch.nn.functional as F


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(pred, target))


def max_abs_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().max()


def relative_l1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = (pred - target).abs().mean()
    denominator = target.abs().mean().clamp_min(eps)
    return numerator / denominator


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.sqrt(((pred - target) ** 2).mean())
    denominator = torch.sqrt((target ** 2).mean()).clamp_min(eps)
    return numerator / denominator