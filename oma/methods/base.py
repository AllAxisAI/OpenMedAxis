from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, List, Iterable, Set

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
        ignore_hparams_list: Optional[Sequence[str]] = None, # Placeholder for future hparams saving control TODO
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


        self.default_ignore = ["model", "loss_fn", "optimizer", "scheduler", "metrics", "inferer"]
        #add ignore_hparams_list to default_ignore
        if ignore_hparams_list is not None:
            self.default_ignore += list(ignore_hparams_list)
        if save_hparams:
            self.save_hyperparameters(
                ignore= self.default_ignore
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
            Ensure the output of `step()` has a consistent format with default values.
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

        if "losses" not in normalized or normalized["losses"] is None:
            normalized["losses"] = {}

        if "state" not in normalized or normalized["state"] is None:
            normalized["state"] = {}

        return normalized

    def _batch_size_from_batch(self, batch: Any) -> Optional[int]:
        """
        Best-effort batch size inference.
        Subclasses can override for structured batch formats.
        """
        if isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        if torch.is_tensor(batch) and batch.ndim > 0:
            return int(batch.shape[0])
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            first = batch[0]
            if torch.is_tensor(first) and first.ndim > 0:
                return int(first.shape[0])
        return None
    
    def _log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        on_step: bool,
        on_epoch: bool,
        prog_bar: bool,
        sync_dist: bool = False,
        batch_size: Optional[int] = None,
    ) -> None:
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
                batch_size=batch_size,
            )

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self._normalize_step_output(self.step(batch, stage="train", batch_idx=batch_idx))
        loss = outputs["loss"]

        if loss is None:
            raise ValueError(
                "`step(..., stage='train')` must return a non-None `loss`."
            )
        
        batch_size = self._batch_size_from_batch(batch)

        self._log_metrics(
            outputs["metrics"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self._normalize_step_output(self.step(batch, stage="val", batch_idx=batch_idx))

        batch_size = self._batch_size_from_batch(batch)
        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )

        return outputs

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self._normalize_step_output(self.step(batch, stage="test", batch_idx=batch_idx))

        batch_size = self._batch_size_from_batch(batch)
        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
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


##############################################
# Advanced users can also use the following lower-level version for more control:
##############################################

class GroupedLossMethod_legacy(Method):
    """
    Extension of OMA Method for grouped-loss training.

    Supports:
    - single-group automatic optimization
    - multi-group manual optimization
    - optimization mode inference
    - reusable optimizer-group stepping conventions

    Expected step() output format:
        {
            "loss": Tensor | None,
            "losses": {
                "main": Tensor,
                "disc": Tensor,
                ...
            },
            "metrics": dict,
            "artifacts": dict,
            "state": dict,
        }

    Notes
    -----
    - If `losses` is empty, `loss` is used as the primary optimization target.
    - If multiple loss groups are present, manual optimization is used by default.
    """

    DEFAULT_OPTIMIZER_GROUP_ORDER: Sequence[str] = ("main", "disc")

    def __init__(
        self,
        *args: Any,
        optimization_mode: str = "infer",   # "infer", "auto", "manual"
        optimizer_group_order: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if optimization_mode not in {"infer", "auto", "manual"}:
            raise ValueError(
                f"Unsupported optimization_mode={optimization_mode}. "
                "Use 'infer', 'auto', or 'manual'."
            )

        self.optimization_mode = optimization_mode
        self.optimizer_group_order = tuple(
            optimizer_group_order or self.DEFAULT_OPTIMIZER_GROUP_ORDER
        )

        # Default value. Final effective behavior is resolved lazily from step output.
        self.automatic_optimization = self.optimization_mode == "auto"

    # ------------------------------------------------------------------
    # normalization
    # ------------------------------------------------------------------
    def _normalize_step_output(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        normalized = super()._normalize_step_output(outputs)

        if "losses" not in normalized or normalized["losses"] is None:
            normalized["losses"] = {}

        if "state" not in normalized or normalized["state"] is None:
            normalized["state"] = {}

        return normalized


    def _extract_loss_dict(self, outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Normalize grouped loss representation.

        If outputs['losses'] is present and non-empty, use it.
        Otherwise, fall back to outputs['loss'] as {'main': loss}.
        """
        losses = outputs.get("losses", {}) or {}
        loss = outputs.get("loss", None)

        if losses:
            cleaned = {}
            for key, value in losses.items():
                if value is None:
                    continue
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"Loss group '{key}' must be a torch.Tensor, got {type(value)}"
                    )
                cleaned[key] = value
            return cleaned

        if loss is None:
            return {}

        if not torch.is_tensor(loss):
            raise TypeError(f"`loss` must be a torch.Tensor, got {type(loss)}")

        return {"main": loss}

    def _active_loss_groups(self, outputs: Dict[str, Any]) -> List[str]:
        return list(self._extract_loss_dict(outputs).keys())

    def _resolve_use_automatic_optimization(self, outputs: Dict[str, Any]) -> bool:
        if self.optimization_mode == "auto":
            return True
        if self.optimization_mode == "manual":
            return False

        active_groups = self._active_loss_groups(outputs)
        return active_groups == ["main"] or len(active_groups) <= 1

    def _ordered_active_groups(self, outputs: Dict[str, Any]) -> List[str]:
        active = self._active_loss_groups(outputs)

        ordered = [g for g in self.optimizer_group_order if g in active]
        unordered_rest = [g for g in active if g not in ordered]
        return ordered + unordered_rest

    def _optimizer_index_for_group(self, group: str, outputs: Dict[str, Any]) -> int:
        ordered_groups = self._ordered_active_groups(outputs)
        if group not in ordered_groups:
            raise KeyError(f"Loss group '{group}' not found in active groups {ordered_groups}")
        return ordered_groups.index(group)

    def _get_optimizer_for_group(self, group: str, outputs: Dict[str, Any]):
        idx = self._optimizer_index_for_group(group, outputs)
        optimizers = self.optimizers()

        if isinstance(optimizers, (list, tuple)):
            if idx >= len(optimizers):
                raise IndexError(
                    f"Optimizer index {idx} for group '{group}' is out of range. "
                    f"Number of optimizers: {len(optimizers)}"
                )
            return optimizers[idx]

        if idx != 0:
            raise IndexError(
                f"Only one optimizer exists, but group '{group}' mapped to index {idx}."
            )
        return optimizers

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------
    def should_step_optimizer(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> bool:
        """
        Hook for subclasses to control update schedules.

        Examples:
        - update discriminator every 2 steps
        - freeze some groups during warmup
        """
        return True

    def before_backward_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    def after_backward_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    def before_optimizer_step_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    def after_optimizer_step_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # manual optimization core
    # ------------------------------------------------------------------
    def manual_step_loss_groups(
        self,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss_dict = self._extract_loss_dict(outputs)
        ordered_groups = self._ordered_active_groups(outputs)

        for group in ordered_groups:
            loss = loss_dict[group]

            if not self.should_step_optimizer(group, loss, outputs, batch, batch_idx):
                continue

            optimizer = self._get_optimizer_for_group(group, outputs)

            optimizer.zero_grad()
            self.before_backward_for_group(group, loss, outputs, optimizer)
            self.manual_backward(loss)
            self.after_backward_for_group(group, loss, outputs, optimizer)
            self.before_optimizer_step_for_group(group, loss, outputs, optimizer)
            optimizer.step()
            self.after_optimizer_step_for_group(group, loss, outputs, optimizer)

    # ------------------------------------------------------------------
    # default steps
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self._normalize_step_output(self.step(batch, stage="train", batch_idx=batch_idx))
        batch_size = self._batch_size_from_batch(batch)

        self._log_metrics(
            outputs["metrics"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )

        use_auto = self._resolve_use_automatic_optimization(outputs)
        self.automatic_optimization = use_auto

        if use_auto:
            loss_dict = self._extract_loss_dict(outputs)
            if "main" in loss_dict:
                return loss_dict["main"]

            if outputs["loss"] is not None:
                return outputs["loss"]

            raise ValueError(
                "Automatic optimization requires either outputs['loss'] "
                "or a 'main' entry in outputs['losses']."
            )

        self.manual_step_loss_groups(outputs, batch, batch_idx)

        loss_dict = self._extract_loss_dict(outputs)
        if "main" in loss_dict:
            return loss_dict["main"]

        return outputs["loss"] if outputs["loss"] is not None else None

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self._normalize_step_output(self.step(batch, stage="val", batch_idx=batch_idx))
        batch_size = self._batch_size_from_batch(batch)

        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )
        return outputs

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self._normalize_step_output(self.step(batch, stage="test", batch_idx=batch_idx))
        batch_size = self._batch_size_from_batch(batch)

        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )
        return outputs



# TODO: Refactor grouped optimizer configuration into GroupedLossMethod
#
# Currently, group-specific optimizer configs (e.g., for "disc") are passed via
# `extra_kwargs` (e.g., "disc_optimizer_cfg"), which is not clean or explicit.
#
# Instead, GroupedLossMethod should natively support:
#   - `optimizer_cfg` for the "main" group
#   - `group_optimizer_cfgs: Dict[str, Dict]` for additional groups (e.g., "disc")
#
# This allows a unified and reusable way to define optimizers for all loss groups.
#
# Example desired API:
#   method = AutoencoderKLMethod(
#       ...,
#       optimizer_cfg={...},  # main
#       group_optimizer_cfgs={
#           "disc": {...},    # discriminator
#       }
#   )
#
# GroupedLossMethod should provide:
#   - get_optimizer_cfg_for_group(group)
#   - build_optimizer_from_cfg(params, cfg)
#
# Then child methods (e.g., AutoencoderKLMethod) only define parameter groups,
# not how optimizers are configured.
#
# This improves:
#   - consistency across methods (GAN, AE, etc.)
#   - separation of concerns
#   - clarity of public API


class GroupedLossMethod(Method):
    """
    Researcher-friendly grouped-loss base.

    Philosophy
    ----------
    - User mainly overrides `build_state(...)`
    - LossComposer computes grouped losses from that state
    - GroupedLossMethod handles:
        * group inference
        * per-group training phases
        * optimizer routing
        * logging
        * scheduling hooks
    - Manual optimization only
    """

    DEFAULT_OPTIMIZER_GROUP_ORDER: Sequence[str] = ("main", "disc")

    def __init__(
        self,
        *args: Any,
        optimizer_group_order: Optional[Sequence[str]] = None,
        group_optimizer_cfgs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.optimizer_group_order = tuple(
            optimizer_group_order or self.DEFAULT_OPTIMIZER_GROUP_ORDER
        )
        self.group_optimizer_cfgs = group_optimizer_cfgs or {}

        # GroupedLossMethod is explicitly manual optimization.
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # researcher-facing hooks
    # ------------------------------------------------------------------
    def build_state(
        self,
        batch: Any,
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main override point for researchers.

        This is where the method defines forward behavior and constructs the
        shared state consumed by the LossComposer.

        The user is free to:
        - run different forwards depending on group
        - skip expensive branches
        - use helper functions
        - return any tensors needed by loss terms

        Return value
        ------------
        dict
            The state dict consumed by self.loss_fn(state).
        """
        raise NotImplementedError

    def prepare_state(
        self,
        state: Dict[str, Any],
        group: Optional[str],
        stage: str,
        batch: Any,
        batch_idx: int,
    ) -> Dict[str, Any]:
        """
        Optional hook to mutate/augment state before loss computation.

        Useful for:
        - adding cached tensors
        - conditional concatenation
        - custom discriminator inputs
        - awkward research-specific plumbing
        """
        return state

    def configure_group_trainability(
        self,
        group: str,
        batch: Any,
        batch_idx: int,
        stage: str,
    ) -> None:
        """
        Optional hook to freeze/unfreeze modules before this group's forward.
        Default: no-op.
        """
        return None

    def should_step_optimizer(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> bool:
        """
        Scheduling hook.

        Examples:
        - skip discriminator warmup
        - update a group every N steps
        """
        return True

    def before_backward_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    def after_backward_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    def before_optimizer_step_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    def after_optimizer_step_for_group(
        self,
        group: str,
        loss: torch.Tensor,
        outputs: Dict[str, Any],
        optimizer: Any,
    ) -> None:
        pass

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    @staticmethod
    def set_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad_(flag)

    @staticmethod
    def _unique_params_from_modules(modules: Iterable[nn.Module]) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: Set[int] = set()
        for module in modules:
            for p in module.parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    params.append(p)
        return params

    # ------------------------------------------------------------------
    # group inference
    # ------------------------------------------------------------------
    def infer_active_groups(self) -> List[str]:
        """
        Infer groups automatically.

        Priority:
        1. loss_fn.groups()
        2. optimizer config keys
        3. fallback ['main']
        """
        if hasattr(self.loss_fn, "groups") and callable(self.loss_fn.groups):
            groups = list(self.loss_fn.groups())
        elif self.group_optimizer_cfgs:
            groups = ["main"] + list(self.group_optimizer_cfgs.keys())
        else:
            groups = ["main"]

        ordered = [g for g in self.optimizer_group_order if g in groups]
        ordered += [g for g in groups if g not in ordered]
        return ordered

    # ------------------------------------------------------------------
    # parameter routing
    # ------------------------------------------------------------------
    def infer_modules_for_group(self, group: str) -> List[nn.Module]:
        """
        Default module inference.

        - 'main' -> self.model
        - others -> nothing by default

        Override for custom groups if needed.
        """
        if group == "main":
            return [self.model] if self.model is not None else []
        return []

    def parameters_for_group(self, group: str) -> Iterable[nn.Parameter]:
        """
        Default parameter routing.

        - 'main' -> self.model.parameters()
        - other groups -> infer_modules_for_group(group)
        """
        if group == "main":
            if self.model is None:
                raise ValueError("GroupedLossMethod requires `self.model` for group='main'.")
            return self.model.parameters()

        modules = self.infer_modules_for_group(group)
        if modules:
            return self._unique_params_from_modules(modules)

        raise NotImplementedError(
            f"Could not infer parameters for group '{group}'. "
            f"Override infer_modules_for_group('{group}') or parameters_for_group('{group}')."
        )

    # ------------------------------------------------------------------
    # optimizer config
    # ------------------------------------------------------------------
    def get_optimizer_cfg_for_group(self, group: str) -> Dict[str, Any]:
        if group == "main":
            if not self.optimizer_cfg:
                raise ValueError("No optimizer_cfg defined for 'main'.")
            return self.optimizer_cfg

        if group not in self.group_optimizer_cfgs:
            raise KeyError(
                f"No optimizer config found for group '{group}'. "
                f"Provide group_optimizer_cfgs['{group}']."
            )
        return self.group_optimizer_cfgs[group]

    def build_optimizer_from_cfg(
        self,
        params: Iterable[nn.Parameter],
        cfg: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        optimizer_cls = cfg.get("class", None)
        optimizer_params = cfg.get("params", {})

        if optimizer_cls is None:
            raise ValueError("Optimizer config must contain a `class` entry.")

        return optimizer_cls(params, **optimizer_params)

    def configured_groups(self) -> List[str]:
        return self.infer_active_groups()

    def _optimizer_index_for_group_name(self, group: str) -> int:
        groups = self.configured_groups()
        if group not in groups:
            raise KeyError(f"Group '{group}' not found in configured groups {groups}.")
        return groups.index(group)

    def _get_optimizer_for_group_name(self, group: str):
        idx = self._optimizer_index_for_group_name(group)
        optimizers = self.optimizers()

        if isinstance(optimizers, (list, tuple)):
            if idx >= len(optimizers):
                raise IndexError(
                    f"Optimizer index {idx} for group '{group}' is out of range. "
                    f"Number of optimizers: {len(optimizers)}"
                )
            return optimizers[idx]

        if idx != 0:
            raise IndexError(
                f"Only one optimizer exists, but group '{group}' mapped to index {idx}."
            )
        return optimizers

    def configure_optimizers(self) -> Any:
        groups = self.configured_groups()
        optimizers = []

        for group in groups:
            params = list(self.parameters_for_group(group))
            if len(params) == 0:
                raise ValueError(f"No parameters found for optimizer group '{group}'.")

            cfg = self.get_optimizer_cfg_for_group(group)
            optimizers.append(self.build_optimizer_from_cfg(params, cfg))

        return optimizers if len(optimizers) > 1 else optimizers[0]

    # ------------------------------------------------------------------
    # engine-owned step
    # ------------------------------------------------------------------
    def _package_outputs(
        self,
        *,
        state: Dict[str, Any],
        loss_outputs: Dict[str, Any],
        group: Optional[str],
    ) -> Dict[str, Any]:
        losses = loss_outputs.get("losses", {}) or {}
        logs = loss_outputs.get("logs", {}) or {}
        term_outputs = loss_outputs.get("term_outputs", {}) or {}

        if group is None:
            selected_loss = losses.get("main", None)
        else:
            if group not in losses:
                raise KeyError(
                    f"LossComposer did not produce requested group '{group}'. "
                    f"Available groups: {list(losses.keys())}"
                )
            selected_loss = losses[group]

        method_metrics = state.pop("_method_metrics", None) or {}
        artifacts = state.pop("_artifacts", None) or {}

        # ----------------------------------------------------------
        # Filter logs for grouped training:
        # only log metrics belonging to the active group.
        # For eval (group=None), keep everything.
        # ----------------------------------------------------------
        if group is None:
            filtered_logs = dict(logs)
        else:
            filtered_logs = {}

            # keep logs only from terms that belong to this group
            for term_name, term_out in term_outputs.items():
                term_group = getattr(term_out, "group", None)
                term_logs = getattr(term_out, "logs", {}) or {}

                if term_group == group:
                    for k, v in term_logs.items():
                        filtered_logs[k] = v

            # also keep the aggregated loss log for this group
            split = state.get("split", "train")
            group_loss_key = f"{split}/{group}_loss"
            if group_loss_key in logs:
                filtered_logs[group_loss_key] = logs[group_loss_key]

        metrics = {}
        metrics.update(method_metrics)
        metrics.update(filtered_logs)

        return self._normalize_step_output(
            {
                "loss": selected_loss,
                "losses": losses,
                "metrics": metrics,
                "artifacts": artifacts,
                "state": state,
                "term_outputs": term_outputs,
            }
        )

    def step(
        self,
        batch: Any,
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Engine-owned step.

        Researchers generally override `build_state(...)`, not this.
        """
        state = self.build_state(batch, stage=stage, batch_idx=batch_idx, group=group)
        if not isinstance(state, dict):
            raise TypeError(f"build_state(...) must return a dict, got {type(state)}")

        state = dict(state)
        state.setdefault("split", stage)
        state.setdefault("global_step", int(self.global_step))

        state = self.prepare_state(
            state=state,
            group=group,
            stage=stage,
            batch=batch,
            batch_idx=batch_idx,
        )

        if self.loss_fn is None:
            raise ValueError("GroupedLossMethod requires `loss_fn`, typically a LossComposer.")

        loss_outputs = self.loss_fn(state, group=group)
        if not isinstance(loss_outputs, dict):
            raise TypeError(f"loss_fn(state) must return a dict, got {type(loss_outputs)}")

        return self._package_outputs(
            state=state,
            loss_outputs=loss_outputs,
            group=group,
        )

    # ------------------------------------------------------------------
    # training / eval
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int):
        batch_size = self._batch_size_from_batch(batch)
        returned_loss = None

        for group in self.infer_active_groups():
            self.configure_group_trainability(
                group=group,
                batch=batch,
                batch_idx=batch_idx,
                stage="train",
            )

            outputs = self.step(batch, stage="train", batch_idx=batch_idx, group=group)
            loss = outputs["loss"]

            if loss is None:
                raise ValueError(
                    f"Grouped training requires a scalar loss for group '{group}'."
                )

            self._log_metrics(
                outputs["metrics"],
                on_step=True,
                on_epoch=True,
                # prog_bar=(group == "main"),
                prog_bar=True,
                sync_dist=False,
                batch_size=batch_size,
            )

            if not self.should_step_optimizer(group, loss, outputs, batch, batch_idx):
                if group == "main" and returned_loss is None:
                    returned_loss = loss
                continue

            if not loss.requires_grad:
                continue

            optimizer = self._get_optimizer_for_group_name(group)

            optimizer.zero_grad()

            self.before_backward_for_group(group, loss, outputs, optimizer)
            self.manual_backward(loss)
            self.after_backward_for_group(group, loss, outputs, optimizer)

            self.before_optimizer_step_for_group(group, loss, outputs, optimizer)
            optimizer.step()
            self.after_optimizer_step_for_group(group, loss, outputs, optimizer)

            optimizer.zero_grad()

            if group == "main" and returned_loss is None:
                returned_loss = loss

        return returned_loss

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self.step(batch, stage="val", batch_idx=batch_idx, group=None)
        batch_size = self._batch_size_from_batch(batch)

        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )
        return outputs

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        outputs = self.step(batch, stage="test", batch_idx=batch_idx, group=None)
        batch_size = self._batch_size_from_batch(batch)

        self._log_metrics(
            outputs["metrics"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch_size,
        )
        return outputs