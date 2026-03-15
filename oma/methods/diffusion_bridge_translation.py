from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F

from .base import Method


class DiffusionBridgeTranslationMethod(Method):
    """
    Non-adversarial diffusion bridge method for paired image translation.

    Expected batch format:
        {
            "source": ...,
            "target": ...,
            "meta": {...},   # optional
        }

    or legacy tuple/list format:
        (target, source)
        (target, source, meta)

    Notes
    -----
    In this method:
    - source is the conditioning image (y in your old code)
    - target is the ground-truth image to reconstruct (x0 in your old code)

    Core training flow:
    1. sample timestep t
    2. sample x_t from q(x_t | x0, y)
    3. recursively predict x0 from (x_t, y, t)
    4. compute reconstruction loss against x0
    """

    def __init__(
        self,
        bridge_model,
        diffusion,
        lambda_rec_loss: float = 1.0,
        n_recursions: int | None = None,
        val_loss_fn=None,
        metrics: Dict[str, Any] | None = None,
        optimizer=None,
        scheduler=None,
        optimizer_cfg: Dict[str, Any] | None = None,
        scheduler_cfg: Dict[str, Any] | None = None,
        save_hparams: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=bridge_model,
            loss_fn=None,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            metrics=metrics,
            inferer=None,
            save_hparams=save_hparams,
            **kwargs,
        )


        self.diffusion = diffusion
        self.lambda_rec_loss = lambda_rec_loss

        # Prefer explicit argument, otherwise fall back to diffusion attribute if present.
        if n_recursions is None:
            n_recursions = getattr(diffusion, "n_recursions", 1)
        self.n_recursions = n_recursions

        # Optional validation/test loss override
        self.val_loss_fn = val_loss_fn or F.mse_loss

        # Read n_steps from diffusion object
        if not hasattr(diffusion, "n_steps"):
            raise AttributeError(
                "`diffusion` must have attribute `n_steps` for timestep sampling."
            )
        self.n_steps = diffusion.n_steps

    def parse_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            target, source, meta

        OMA preferred dict format:
            {
                "source": conditioning image,
                "target": reconstruction target,
                "meta": ...
            }

        Legacy tuple support:
            (target, source)
            (target, source, meta)
        """
        if isinstance(batch, dict):
            if "source" not in batch or "target" not in batch:
                raise KeyError(
                    "Dictionary batch must contain 'source' and 'target' keys."
                )
            source = batch["source"]
            target = batch["target"]
            meta = batch.get("meta", {})
            return target, source, meta

        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                target, source = batch
                return target, source, {}

            if len(batch) == 3:
                target, source, meta = batch
                return target, source, meta if meta is not None else {}

            raise ValueError(
                "Tuple/list batch must have length 2 or 3 for bridge translation."
            )

        raise TypeError(
            "Unsupported batch type for DiffusionBridgeTranslationMethod."
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.n_steps + 1, (batch_size,), device=device)

    def predict_x0(self, x_t: torch.Tensor, source: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Recursive x0 prediction using the bridge model.
        """
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.model(
                torch.cat((x_t, source), dim=1),
                t,
                x_r=x0_r,
            )
        return x0_r

    def infer(self, source: torch.Tensor) -> torch.Tensor:
        """
        Validation/test inference: full sampling from the diffusion bridge.
        """
        return self.diffusion.sample_x0(source, self.model)

    def compute_train_loss(
        self,
        x0_pred: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        return self.lambda_rec_loss * F.l1_loss(x0_pred, x0, reduction="sum")

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        stage: str,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if not self.metrics:
            return results

        for name, metric_fn in self.metrics.items():
            results[f"{stage}/{name}"] = metric_fn(pred, target)

        return results

    def step(self, batch: Any, stage: str, batch_idx: int) -> Dict[str, Any]:
        x0, source, meta = self.parse_batch(batch)

        if stage == "train":
            t = self.sample_timesteps(x0.shape[0], x0.device)
            x_t = self.diffusion.q_sample(t, x0, source)
            x0_pred = self.predict_x0(x_t, source, t)

            loss = self.compute_train_loss(x0_pred, x0)

            metrics = {
                f"{stage}/loss": loss,
            }
            metrics.update(self.compute_metrics(x0_pred, x0, stage=stage))

            return {
                "loss": loss,
                "metrics": metrics,
                "artifacts": {
                    "source": source,
                    "target": x0,
                    "pred": x0_pred,
                    "t": t,
                    "meta": meta,
                },
            }

        # val / test path
        x0_pred = self.infer(source)
        loss = self.val_loss_fn(x0_pred, x0)

        metrics = {
            f"{stage}/loss": loss,
        }
        metrics.update(self.compute_metrics(x0_pred, x0, stage=stage))

        step_out = {
            "loss": loss,
            "metrics": metrics,
            "artifacts": {
                "source": source,
                "target": x0,
                "pred": x0_pred,
                "meta": meta,
            },
        }

        print(f"Evaluator exists: {self.evaluator_manager is not None}")
        if self.evaluator_manager is not None:

            eval_results = self.evaluator_manager.run(
                stage=stage,
                outputs=step_out["artifacts"],
                step=batch_idx,
                output_dir=self.logger.log_dir
            )
            
            step_out["eval_results"] = eval_results

        return step_out