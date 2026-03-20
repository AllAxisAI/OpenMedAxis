from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .base import GroupedLossMethod


class AutoencoderKLMethod(GroupedLossMethod):
    """
    OMA method for AutoencoderKL-style models.

    Expected model API
    ------------------
    The model should support:

        model(x, sample_posterior=True, return_latent=True)

    and return either:
        - (recon, posterior, latent)
        - (recon, posterior)

    Optional model API:
        - get_last_layer()

    Expected loss API
    -----------------
    self.loss_fn(state) should return:

        {
            "losses": {"main": ..., "disc": ...},
            "logs": {...},
            "term_outputs": {...},
        }

    The GroupedLossMethod base handles grouped manual optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        image_key: str = "image",
        sample_posterior: bool = True,
        save_validation_artifacts: bool = True,
        val_artifact_first_batch_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            **kwargs,
        )
        self.image_key = image_key
        self.sample_posterior = sample_posterior

        self.save_validation_artifacts = save_validation_artifacts
        self.val_artifact_first_batch_only = val_artifact_first_batch_only
        self.validation_artifacts: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # batch parsing
    # ------------------------------------------------------------------
    def parse_batch(self, batch: Any) -> torch.Tensor:
        if not isinstance(batch, dict):
            raise TypeError(
                f"{self.__class__.__name__} expects batch to be a dict, got {type(batch)}"
            )

        if self.image_key not in batch:
            raise KeyError(
                f"Batch does not contain image key '{self.image_key}'. "
                f"Available keys: {list(batch.keys())}"
            )

        x = batch[self.image_key]

        if not torch.is_tensor(x):
            raise TypeError(
                f"Batch key '{self.image_key}' must be a torch.Tensor, got {type(x)}"
            )

        # [B, H, W] -> [B, 1, H, W]
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # [B, H, W, C] -> [B, C, H, W]
        elif x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        if x.ndim != 4:
            raise ValueError(
                f"Expected image tensor with shape [B,C,H,W] or [B,H,W]/[B,H,W,C], "
                f"got shape {tuple(x.shape)}"
            )

        return x.contiguous().float()

    def _batch_size_from_batch(self, batch: Any) -> Optional[int]:
        if isinstance(batch, dict) and self.image_key in batch:
            x = batch[self.image_key]
            if torch.is_tensor(x) and x.ndim > 0:
                return int(x.shape[0])
        return super()._batch_size_from_batch(batch)

    # ------------------------------------------------------------------
    # model helpers
    # ------------------------------------------------------------------
    def _forward_model(self, x: torch.Tensor):
        out = self.model(
            x,
            sample_posterior=self.sample_posterior,
            return_latent=True,
        )

        if not isinstance(out, (tuple, list)):
            raise TypeError(
                "Autoencoder model must return tuple/list like "
                "(recon, posterior, latent) or (recon, posterior)."
            )

        if len(out) == 3:
            recon, posterior, latent = out
        elif len(out) == 2:
            recon, posterior = out
            latent = None
        else:
            raise ValueError(
                f"Unexpected number of outputs from model: {len(out)}"
            )

        return recon, posterior, latent

    def build_state(
        self,
        batch: Any,
        stage: str,
        batch_idx: int,
    ) -> Dict[str, Any]:
        x = self.parse_batch(batch)
        recon, posterior, latent = self._forward_model(x)

        state: Dict[str, Any] = {
            "input": x,
            "recon": recon,
            "posterior": posterior,
            "latent": latent,
            "batch": batch,
            "batch_idx": batch_idx,
            "global_step": int(self.global_step),
            "split": stage,
        }

        if hasattr(self.model, "get_last_layer") and callable(self.model.get_last_layer):
            try:
                state["last_layer"] = self.model.get_last_layer()
            except Exception:
                pass

        return state

    # ------------------------------------------------------------------
    # oma step
    # ------------------------------------------------------------------
    def step(self, batch: Any, stage: str, batch_idx: int) -> Dict[str, Any]:
        state = self.build_state(batch=batch, stage=stage, batch_idx=batch_idx)
        loss_out = self.loss_fn(state)

        if not isinstance(loss_out, dict):
            raise TypeError(
                f"loss_fn(state) must return a dict, got {type(loss_out)}"
            )

        losses = loss_out.get("losses", {})
        metrics = loss_out.get("logs", {})
        term_outputs = loss_out.get("term_outputs", {})

        main_loss = losses.get("main", None)
        if main_loss is None and "loss" in loss_out:
            main_loss = loss_out["loss"]

        artifacts = {
            "input": state["input"],
            "recon": state["recon"],
            "latent": state.get("latent", None),
            "posterior": state.get("posterior", None),
        }

        if stage == "val" and self.save_validation_artifacts:
            if (not self.val_artifact_first_batch_only) or batch_idx == 0:
                self.validation_artifacts = artifacts
            
            # print(f"Evaluator exists: {self.evaluator_manager is not None}")
            if self.evaluator_manager is not None:

                eval_results = self.evaluator_manager.run(
                    stage=stage,
                    outputs=artifacts,
                    step=batch_idx,
                    output_dir=self.logger.log_dir
                )


        return {
            "loss": main_loss,
            "losses": losses,
            "metrics": metrics,
            "artifacts": artifacts,
            "state": state,
            "term_outputs": term_outputs,
        }

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,
    ) -> Dict[str, Any]:
        out = self.model(
            x,
            sample_posterior=sample_posterior,
            return_latent=True,
        )

        if not isinstance(out, (tuple, list)):
            raise TypeError(
                "Autoencoder model must return tuple/list like "
                "(recon, posterior, latent) or (recon, posterior)."
            )

        if len(out) == 3:
            recon, posterior, latent = out
        elif len(out) == 2:
            recon, posterior = out
            latent = None
        else:
            raise ValueError(
                f"Unexpected number of outputs from model: {len(out)}"
            )

        return {
            "recon": recon,
            "posterior": posterior,
            "latent": latent,
        }

    # ------------------------------------------------------------------
    # optimizers
    # ------------------------------------------------------------------
    def _main_parameters(self):
        return self.model.parameters()

    def _disc_parameters(self):
        if not hasattr(self.loss_fn, "terms"):
            return []

        params = []
        seen = set()

        for term in self.loss_fn.terms:
            disc = getattr(term, "discriminator", None)
            if disc is None:
                continue

            for p in disc.parameters():
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    params.append(p)

        return params

    def configure_optimizers(self):
        """
        Uses the base Method config style for main optimizer if provided.

        For grouped-loss training with discriminator terms:
        - optimizer 0 -> main/model params
        - optimizer 1 -> discriminator params
        """
        # If user explicitly passed a ready-made optimizer, respect base behavior
        # only when there is no discriminator branch.
        has_disc = False
        if hasattr(self.loss_fn, "groups") and callable(self.loss_fn.groups):
            has_disc = "disc" in self.loss_fn.groups()

        if not has_disc:
            return super().configure_optimizers()

        # grouped case: create optimizers explicitly
        if not self.optimizer_cfg:
            raise ValueError(
                "Grouped autoencoder training with discriminator requires "
                "`optimizer_cfg` or overriding `configure_optimizers()`."
            )

        optimizer_cls = self.optimizer_cfg.get("class", None)
        optimizer_params = self.optimizer_cfg.get("params", {})

        if optimizer_cls is None:
            raise ValueError("`optimizer_cfg` must contain a `class` entry.")

        main_opt = optimizer_cls(self._main_parameters(), **optimizer_params)

        disc_params = self._disc_parameters()
        if len(disc_params) == 0:
            raise ValueError(
                "Loss groups include 'disc' but no discriminator parameters were found."
            )

        disc_optimizer_cfg = self.extra_kwargs.get("disc_optimizer_cfg", None)
        if disc_optimizer_cfg is None:
            disc_optimizer_cfg = self.optimizer_cfg

        disc_optimizer_cls = disc_optimizer_cfg.get("class", None)
        disc_optimizer_params = disc_optimizer_cfg.get("params", {})

        if disc_optimizer_cls is None:
            raise ValueError("`disc_optimizer_cfg` must contain a `class` entry.")

        disc_opt = disc_optimizer_cls(disc_params, **disc_optimizer_params)

        # Scheduler support can be added later for grouped optimizers if needed.
        return [main_opt, disc_opt]

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def get_validation_artifacts(self) -> Optional[Dict[str, Any]]:
        return self.validation_artifacts