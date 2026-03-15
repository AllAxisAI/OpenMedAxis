from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .base import Method


class TranslationMethod(Method):
    """
    Base method for paired image-to-image translation tasks.

    Expected conceptual batch format:
        {
            "source": ...,
            "target": ...,
            "meta": {...},   # optional
        }

    For migration convenience, tuple/list batches are also supported:
        (source, target)
        (source, target, meta)

    Main flow:
        source -> model -> pred
        pred vs target -> loss
        pred vs target -> metrics

    Subclasses may override:
        - parse_batch()
        - infer()
        - compute_loss()
        - compute_metrics()
        - step() if they need fuller control
    """

    def parse_batch(self, batch: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Parse batch into (source, target, meta).

        Supported formats:
        1. dict with keys "source", "target", optional "meta"
        2. tuple/list of length 2: (source, target)
        3. tuple/list of length 3: (source, target, meta)
        """
        if isinstance(batch, dict):
            if "source" not in batch or "target" not in batch:
                raise KeyError(
                    "Dictionary batch for TranslationMethod must contain "
                    "'source' and 'target' keys."
                )
            source = batch["source"]
            target = batch["target"]
            meta = batch.get("meta", {})
            return source, target, meta

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                source, target = batch
                meta = {}
                return source, target, meta

            if len(batch) == 3:
                source, target, meta = batch
                if meta is None:
                    meta = {}
                return source, target, meta

            raise ValueError(
                "Tuple/list batch for TranslationMethod must have length 2 or 3."
            )

        raise TypeError(
            "Unsupported batch type for TranslationMethod. Expected dict, tuple, or list."
        )

    def infer(self, source: Any) -> Any:
        """
        Forward prediction from source.

        Override this if the method needs custom inference behavior.
        """
        return self(source)

    def compute_loss(self, pred: Any, target: Any, stage: str) -> torch.Tensor:
        """
        Compute the main supervised loss.

        By default this uses `self.loss_fn`, which must be provided unless
        a subclass overrides this method.
        """
        if self.loss_fn is None:
            raise ValueError(
                "`self.loss_fn` is None. Provide a loss_fn or override `compute_loss()`."
            )
        return self.loss_fn(pred, target)

    def compute_metrics(
        self,
        pred: Any,
        target: Any,
        source: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
        stage: str = "train",
    ) -> Dict[str, Any]:
        """
        Compute metric dictionary.

        Default behavior:
        - returns an empty dict if no metrics were provided
        - otherwise calls each metric function as metric_fn(pred, target)

        Metric names are automatically prefixed with stage, e.g.:
            train/psnr
            val/ssim
        """
        if not self.metrics:
            return {}

        results: Dict[str, Any] = {}
        for name, metric_fn in self.metrics.items():
            value = metric_fn(pred, target)
            results[f"{stage}/{name}"] = value

        return results

    def step(self, batch: Any, stage: str) -> Dict[str, Any]:
        """
        Standard paired translation step.
        """
        source, target, meta = self.parse_batch(batch)

        pred = self.infer(source)
        loss = self.compute_loss(pred, target, stage=stage)

        metrics = {f"{stage}/loss": loss}
        metrics.update(
            self.compute_metrics(
                pred=pred,
                target=target,
                source=source,
                meta=meta,
                stage=stage,
            )
        )

        return {
            "loss": loss,
            "metrics": metrics,
            "artifacts": {
                "source": source,
                "target": target,
                "pred": pred,
                "meta": meta,
            },
        }