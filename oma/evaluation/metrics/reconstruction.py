from __future__ import annotations

import warnings

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def norm_01(x: np.ndarray) -> np.ndarray:
    denom = x.max(axis=(-1, -2), keepdims=True) - x.min(axis=(-1, -2), keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return (x - x.min(axis=(-1, -2), keepdims=True)) / denom


def mean_norm(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    denom = x.mean(axis=(-1, -2), keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def apply_mask_and_norm(
    x: np.ndarray,
    mask: np.ndarray,
    norm_func,
) -> np.ndarray:
    x = x * mask
    x = norm_func(x)
    return x


def center_crop(x: np.ndarray, crop: tuple[int, int]) -> np.ndarray:
    h, w = x.shape[-2:]
    ch, cw = crop
    return x[..., h // 2 - ch // 2 : h // 2 + ch // 2, w // 2 - cw // 2 : w // 2 + cw // 2]


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _prepare_images(
    gt_images: torch.Tensor | np.ndarray,
    pred_images: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    norm: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    gt_images = _to_numpy(gt_images)
    pred_images = _to_numpy(pred_images)

    if mask is not None:
        mask = _to_numpy(mask)

    # If images are 4D, remove singleton channel dimension when appropriate
    gt_images = gt_images.squeeze() if gt_images.ndim == 4 else gt_images
    pred_images = pred_images.squeeze() if pred_images.ndim == 4 else pred_images

    # If single 2D image, promote to batch of size 1
    gt_images = gt_images[None, ...] if gt_images.ndim == 2 else gt_images
    pred_images = pred_images[None, ...] if pred_images.ndim == 2 else pred_images

    if gt_images.shape != pred_images.shape:
        raise ValueError("Ground truth and predicted images must have the same shape.")

    if norm == "mean":
        norm_func = mean_norm
    elif norm == "01":
        norm_func = norm_01
    else:
        raise ValueError(f"Unsupported norm: {norm}. Expected 'mean' or '01'.")

    # Match old utility behavior: if values look like [-1, 1], remap to [0, 1]
    if np.nanmin(gt_images) < -0.1:
        gt_images = ((gt_images + 1.0) / 2.0).clip(0.0, 1.0)

    if np.nanmin(pred_images) < -0.1:
        pred_images = ((pred_images + 1.0) / 2.0).clip(0.0, 1.0)

    if mask is not None:
        gt_images = center_crop(gt_images, mask.shape[-2:])
        pred_images = center_crop(pred_images, mask.shape[-2:])

        gt_images = apply_mask_and_norm(gt_images, mask, norm_func)
        pred_images = apply_mask_and_norm(pred_images, mask, norm_func)
    else:
        gt_images = norm_func(gt_images)
        pred_images = norm_func(pred_images)

    return gt_images, pred_images


def psnr(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    norm: str = "mean",
) -> torch.Tensor:
    """
    Mean PSNR over batch/slices, following the behavior of the old utility.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        gt_images, pred_images = _prepare_images(
            gt_images=target,
            pred_images=pred,
            mask=mask,
            norm=norm,
        )

        values = []
        for gt, pr in zip(gt_images, pred_images):
            gt = np.squeeze(gt)
            pr = np.squeeze(pr)

            data_range = float(np.max(gt))
            if data_range <= 0 or np.isnan(data_range):
                data_range = 1.0

            value = sk_psnr(gt, pr, data_range=data_range)
            values.append(value)

        mean_value = float(np.nanmean(np.asarray(values, dtype=np.float64)))
        return torch.tensor(mean_value, dtype=torch.float32)


def ssim(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    norm: str = "mean",
    multiply_by_100: bool = True,
) -> torch.Tensor:
    """
    Mean SSIM over batch/slices, following the behavior of the old utility.

    By default multiplies the result by 100, matching your previous file.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        gt_images, pred_images = _prepare_images(
            gt_images=target,
            pred_images=pred,
            mask=mask,
            norm=norm,
        )

        values = []
        for gt, pr in zip(gt_images, pred_images):
            gt = np.squeeze(gt)
            pr = np.squeeze(pr)

            data_range = float(np.max(gt))
            if data_range <= 0 or np.isnan(data_range):
                data_range = 1.0

            value = sk_ssim(gt, pr, data_range=data_range)
            if multiply_by_100:
                value *= 100.0
            values.append(value)

        mean_value = float(np.nanmean(np.asarray(values, dtype=np.float64)))
        return torch.tensor(mean_value, dtype=torch.float32)


class PSNRMetric:
    def __init__(
        self,
        mask: torch.Tensor | np.ndarray | None = None,
        norm: str = "mean",
    ) -> None:
        self.mask = mask
        self.norm = norm

    def __call__(
        self,
        pred: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        return psnr(
            pred=pred,
            target=target,
            mask=self.mask,
            norm=self.norm,
        )


class SSIMMetric:
    def __init__(
        self,
        mask: torch.Tensor | np.ndarray | None = None,
        norm: str = "mean",
        multiply_by_100: bool = True,
    ) -> None:
        self.mask = mask
        self.norm = norm
        self.multiply_by_100 = multiply_by_100

    def __call__(
        self,
        pred: torch.Tensor | np.ndarray,
        target: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        return ssim(
            pred=pred,
            target=target,
            mask=self.mask,
            norm=self.norm,
            multiply_by_100=self.multiply_by_100,
        )