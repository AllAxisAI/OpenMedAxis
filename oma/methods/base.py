from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import lightning as L
import torch
from torch import nn


class Method(L.LightningModule):
    """
    Base training abstraction for OpenMedAxis (OMA).

    Philosophy
    ----------
    - Lightning handles execution/infrastructure.
    - OMA Method defines the algorithmic training interface.
    - Subclasses should mainly override:
        * parse_batch()
        * step()
        * optionally configure_optimizers()

    Expected output format of `step()`:
        {
            "loss": torch.Tensor | None,
            "metrics": dict[str, Any],
            "artifacts": dict[str, Any],
        }
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[Callable[..., Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        optimizer_cfg: Optional[Dict[str, Any]] = None,
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Callable[..., Any]]] = None,
        inferer: Optional[Callable[..., Any]] = None,
        save_hparams: bool = True,
        evaluator_manager: Optional[Any] = None, # Placeholder for future integration with EvaluatorManager TODO
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn

        # Pre-built optimizer / scheduler instances
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Config-driven optimizer / scheduler creation
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}

        # Optional metric callables, left flexible for child methods
        self.metrics = metrics or {}

        self.evaluator_manager = evaluator_manager

        # Optional inference helper for predict_step
        self.inferer = inferer

        # Store any extra kwargs for child classes if needed
        self.extra_kwargs = kwargs

        if save_hparams:
            self.save_hyperparameters(
                ignore=[
                    "model",
                    "loss_fn",
                    "optimizer",
                    "scheduler",
                    "metrics",
                    "inferer",
                ]
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Default forward delegates to `self.model`.

        Override this if the method contains multiple models or custom forward logic.
        """
        if self.model is None:
            raise NotImplementedError(
                "`self.model` is None. Provide a model or override `forward()`."
            )
        return self.model(*args, **kwargs)

    def parse_batch(self, batch: Any) -> Any:
        """
        Parse a raw batch into the format expected by the method.

        Child classes should override this when they need structured access to
        source/target/meta or other task-specific fields.
        """
        return batch

    def step(self, batch: Any, stage: str, batch_index: int) -> Dict[str, Any]:
        """
        Core algorithmic step.

        Must be implemented by subclasses.

        Parameters
        ----------
        batch:
            Raw batch from dataloader.
        stage:
            One of {"train", "val", "test"}.

        Returns
        -------
        dict
            A dictionary with the standard OMA step format:
            {
                "loss": Tensor | None,
                "metrics": dict,
                "artifacts": dict,
            }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.step() must be implemented by subclasses."
        )

    def _normalize_step_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure the step output matches OMA conventions.
        """
        if not isinstance(outputs, dict):
            raise TypeError(
                f"`step()` must return a dict, but got {type(outputs).__name__}."
            )

        normalized = dict(outputs)

        if "loss" not in normalized:
            normalized["loss"] = None

        if "metrics" not in normalized or normalized["metrics"] is None:
            normalized["metrics"] = {}

        if "artifacts" not in normalized or normalized["artifacts"] is None:
            normalized["artifacts"] = {}

        return normalized

    def _log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool,
        sync_dist: bool = False,
    ) -> None:
        """
        Log a metrics dictionary safely through Lightning.
        """
        if not metrics:
            return

        processed: Dict[str, Any] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            processed[key] = value

        if processed:
            self.log_dict(
                processed,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar,
                logger=True,
                sync_dist=sync_dist,
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self._normalize_step_output(self.step(batch, stage="train", batch_idx=batch_idx))
        loss = outputs["loss"]

        if loss is None:
            raise ValueError(
                "`step(..., stage='train')` must return a non-None `loss`."
            )

        self._log_metrics(
            outputs["metrics"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self._normalize_step_output(self.step(batch, stage="val", batch_idx=batch_idx))

        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return outputs

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self._normalize_step_output(self.step(batch, stage="test", batch_idx=batch_idx))

        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        return outputs

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        """
        Default prediction path.

        Order of behavior:
        1. parse batch
        2. use `inferer` if provided
        3. otherwise call forward with best-effort unpacking
        """
        parsed = self.parse_batch(batch)

        if self.inferer is not None:
            return self.inferer(self, parsed)

        if isinstance(parsed, dict):
            return self(**parsed)

        if isinstance(parsed, (tuple, list)):
            return self(*parsed)

        return self(parsed)

    def configure_optimizers(self) -> Any:
        """
        Supports:
        1. prebuilt optimizer / scheduler objects
        2. config-driven creation via optimizer_cfg / scheduler_cfg
        3. full override in subclasses for custom optimization patterns

        Example optimizer_cfg
        ---------------------
        {
            "class": torch.optim.Adam,
            "params": {"lr": 1e-4, "betas": (0.9, 0.999)}
        }

        Example scheduler_cfg
        ---------------------
        {
            "class": torch.optim.lr_scheduler.StepLR,
            "params": {"step_size": 10, "gamma": 0.5},
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss"
        }
        """
        optimizer = self._optimizer
        scheduler = self._scheduler

        if optimizer is None:
            if not self.optimizer_cfg:
                raise ValueError(
                    "No optimizer provided. Pass `optimizer=...`, define "
                    "`optimizer_cfg`, or override `configure_optimizers()`."
                )

            optimizer_cls = self.optimizer_cfg.get("class", None)
            optimizer_params = self.optimizer_cfg.get("params", {})

            if optimizer_cls is None:
                raise ValueError("`optimizer_cfg` must contain a `class` entry.")

            optimizer = optimizer_cls(self.parameters(), **optimizer_params)

        if scheduler is None and self.scheduler_cfg:
            scheduler_cls = self.scheduler_cfg.get("class", None)
            scheduler_params = self.scheduler_cfg.get("params", {})

            if scheduler_cls is None:
                raise ValueError("`scheduler_cfg` must contain a `class` entry.")

            scheduler = scheduler_cls(optimizer, **scheduler_params)

        if scheduler is None:
            return optimizer

        scheduler_dict: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.scheduler_cfg.get("interval", "epoch"),
            "frequency": self.scheduler_cfg.get("frequency", 1),
        }

        if "monitor" in self.scheduler_cfg:
            scheduler_dict["monitor"] = self.scheduler_cfg["monitor"]

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }