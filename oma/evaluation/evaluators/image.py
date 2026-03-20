from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .base import Evaluator, EvaluatorOutput


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def _prepare_batch_images(x: torch.Tensor | np.ndarray) -> np.ndarray:
    x = _to_numpy(x)

    # (B, 1, H, W) -> (B, H, W)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]

    # (H, W) -> (1, H, W)
    if x.ndim == 2:
        x = x[None]

    if x.ndim != 3:
        raise ValueError(
            f"Expected image tensor with shape (B,H,W) or (B,1,H,W), got {x.shape}"
        )

    return x


def _normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin = float(np.min(img))
    vmax = float(np.max(img))

    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    return (img - vmin) / (vmax - vmin)


class SaveImageEvaluator(Evaluator):
    def __init__(
        self,
        name: str = "images",
        max_samples: int = 4,
        save_every_n_steps: int = 1,
        dpi: int = 150,
        output_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.max_samples = max_samples
        self.save_every_n_steps = save_every_n_steps
        self.dpi = dpi
        self._output_dir = output_dir

    def __call__(
        self,
        *,
        stage: str,
        outputs: Mapping[str, Any],
        output_dir: Optional[str] = None,
        step: Optional[int] = None,
    ) -> EvaluatorOutput:
        if output_dir is None:
            # return EvaluatorOutput()
            if self._output_dir is None:
                raise ValueError("Output directory must be specified either in constructor or call.")
            output_dir = self._output_dir


        print(f"SaveImageEvaluator: stage={stage}, step={step}, output_dir={output_dir}")


        if step is not None and self.save_every_n_steps > 1:
            if step % self.save_every_n_steps != 0:
                return EvaluatorOutput()

        source = outputs.get("source")
        target = outputs.get("target")
        pred = outputs.get("pred")

        if source is None or target is None or pred is None:
            raise KeyError(
                "SaveImageEvaluator requires outputs to contain 'source', 'target', and 'pred'."
            )

        source = _prepare_batch_images(source)
        target = _prepare_batch_images(target)
        pred = _prepare_batch_images(pred)

        n = min(self.max_samples, len(pred))

        save_dir = os.path.join(output_dir, stage, self.name)
        if step is not None:
            save_dir = os.path.join(save_dir, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)

        artifact_paths = {}

        for i in range(n):
            src_i = _normalize_for_display(source[i])
            tgt_i = _normalize_for_display(target[i])
            pred_i = _normalize_for_display(pred[i])
            err_i = np.abs(pred_i - tgt_i)

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))

            axes[0].imshow(src_i, cmap="gray")
            axes[0].set_title("Source")
            axes[0].axis("off")

            axes[1].imshow(pred_i, cmap="gray")
            axes[1].set_title("Pred")
            axes[1].axis("off")

            axes[2].imshow(tgt_i, cmap="gray")
            axes[2].set_title("Target")
            axes[2].axis("off")

            axes[3].imshow(err_i, cmap="gray")
            axes[3].set_title("|Pred-Target|")
            axes[3].axis("off")

            fig.tight_layout()

            path = os.path.join(save_dir, f"sample_{i}.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            artifact_paths[f"sample_{i}_path"] = path

        return EvaluatorOutput(artifacts=artifact_paths)

class SaveImageEvaluatorGeneric(Evaluator):
    def __init__(
        self,
        name: str = "images",
        max_samples: int = 4,
        save_every_n_steps: int = 1,
        dpi: int = 150,
        output_dir: Optional[str] = None,
        image_keys: Optional[list[str]] = None,
    ) -> None:
        super().__init__(name=name)
        self.max_samples = max_samples
        self.save_every_n_steps = save_every_n_steps
        self.dpi = dpi
        self._output_dir = output_dir
        self.image_keys = image_keys or ["pred"]

    def __call__(
        self,
        *,
        stage: str,
        outputs: Mapping[str, Any],
        output_dir: Optional[str] = None,
        step: Optional[int] = None,
    ) -> EvaluatorOutput:
        if output_dir is None:
            # return EvaluatorOutput()
            if self._output_dir is None:
                raise ValueError("Output directory must be specified either in constructor or call.")
            output_dir = self._output_dir

        if step is not None and self.save_every_n_steps > 1:
            if step % self.save_every_n_steps != 0:
                return EvaluatorOutput()
        
        print(f"SaveImageEvaluator: stage={stage}, step={step}, output_dir={output_dir}")
        # prepare images
        images = {}
        for key in self.image_keys:
            img = outputs.get(key)
            if img is None:
                raise KeyError(
                    f"SaveImageEvaluatorSingle requires outputs to contain '{key}'."
                )
            images[key] = _prepare_batch_images(img)
        
        n = min(self.max_samples, images[self.image_keys[0]].shape[0])

        save_dir = os.path.join(output_dir, stage, self.name)
        if step is not None:
            save_dir = os.path.join(save_dir, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)

        artifact_paths = {}

        for i in range(n):

            fig, axes = plt.subplots(1, len(self.image_keys), figsize=(4*len(self.image_keys), 4))

            for j, key in enumerate(self.image_keys):
                img_i = _normalize_for_display(images[key][i])
                axes[j].imshow(img_i, cmap="gray")
                axes[j].set_title(key)
                axes[j].axis("off")

            fig.tight_layout()

            path = os.path.join(save_dir, f"sample_{i}.png")
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            artifact_paths[f"sample_{i}_path"] = path

        return EvaluatorOutput(artifacts=artifact_paths)




        