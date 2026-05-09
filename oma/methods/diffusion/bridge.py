from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .base import BaseDiffusionMethod

logger = logging.getLogger(__name__)


class DiffusionBridgeMethod(BaseDiffusionMethod):
    """
    Diffusion bridge method for paired image-to-image translation.

    Wraps a ``BridgeDiffusionProcess`` and a recursive x0-prediction network
    into OMA's ``BaseDiffusionMethod`` / ``GroupedLossMethod`` training loop.

    Expected batch format
    ---------------------
    Dict::

        {
            "source": <conditioning image>,
            "target": <reconstruction target>,
            "meta":   <optional metadata>,
        }

    Tuple/list::

        (target, source)
        (target, source, meta)

    Training flow
    -------------
    1. Parse batch → x0 (target), source (cond), meta
    2. Sample timestep t ~ Uniform[1, n_steps]
    3. Sample x_t ~ q(x_t | x0, source) via BridgeDiffusionProcess
    4. Recursively predict x0 from (x_t, source, t) using the network
    5. Compute loss (default: L1 on x0_pred vs x0)

    Inference
    ---------
    Delegates to ``process.sample_x0(source, model)``, which runs the full
    iterative reverse chain.

    Parameters
    ----------
    process:
        A ``BridgeDiffusionProcess`` instance. Owns the schedule math and
        full reverse sampling.
    model:
        The denoising / bridge network.
        Expected call signature: ``model(cat(x_t, source), t, x_r=x0_r)``.
    loss_fn:
        OMA ``LossComposer`` or compatible callable that maps the state dict
        to a loss dict. If ``None``, defaults to L1 on ``pred`` vs ``target``
        scaled by ``lambda_rec``.
    lambda_rec:
        Weight for the default L1 reconstruction loss.
        Ignored when ``loss_fn`` is provided explicitly.
    n_recursions:
        Recursive refinement steps during training. If ``None``, uses
        ``process.n_recursions``.
    consistency_threshold:
        Early stopping threshold for training recursion. If ``None``, uses
        ``process.consistency_threshold``.

    Examples
    --------
    Minimal setup::

        process = BridgeDiffusionProcess(n_steps=10, beta_start=0.1, beta_end=3.0, gamma=1.0)
        method  = DiffusionBridgeMethod(
            process=process,
            model=bridge_model,
        )

    Custom loss::

        from oma.losses.terms import L1LossTerm
        from oma.losses.composer import LossComposer

        method = DiffusionBridgeMethod(
            process=process,
            model=bridge_model,
            loss_fn=LossComposer([
                L1LossTerm(pred_key="pred", target_key="target", weight=2.0),
            ]),
        )
    """

    def __init__(
        self,
        *args: Any,
        lambda_rec: float = 1.0,
        n_recursions: Optional[int] = None,
        consistency_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # Build a default L1 loss if none provided
        if "loss_fn" not in kwargs or kwargs.get("loss_fn") is None:
            from oma.losses.terms import L1LossTerm
            from oma.losses.composer import LossComposer

            kwargs["loss_fn"] = LossComposer([
                L1LossTerm(
                    pred_key="pred",
                    target_key="target",
                    weight=lambda_rec,
                    name="bridge_l1",
                ),
            ])

        super().__init__(*args, **kwargs)

        self.lambda_rec = float(lambda_rec)
        self._n_recursions_override = n_recursions
        self._consistency_threshold_override = consistency_threshold

    @property
    def n_recursions(self) -> int:
        if self._n_recursions_override is not None:
            return int(self._n_recursions_override)
        return int(getattr(self.process, "n_recursions", 1))

    @property
    def consistency_threshold(self) -> float:
        if self._consistency_threshold_override is not None:
            return float(self._consistency_threshold_override)
        return float(getattr(self.process, "consistency_threshold", 0.0))

    # ------------------------------------------------------------------
    # batch parsing
    # ------------------------------------------------------------------
    def parse_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Supported formats
        -----------------
        Dict:  ``{"source": ..., "target": ..., "meta": ...}``
        Tuple: ``(target, source)`` or ``(target, source, meta)``
        """
        if isinstance(batch, dict):
            if "source" not in batch or "target" not in batch:
                raise KeyError(
                    "DiffusionBridgeMethod dict batch must contain 'source' and 'target'."
                )
            return {
                "x0":        batch["target"],
                "cond":      batch["source"],
                "meta":      batch.get("meta", None),
                "raw_batch": batch,
            }

        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError(
                    "DiffusionBridgeMethod tuple batch must have at least 2 elements: (target, source)."
                )
            return {
                "x0":        batch[0],
                "cond":      batch[1],
                "meta":      batch[2] if len(batch) >= 3 else None,
                "raw_batch": batch,
            }

        raise TypeError(
            f"Unsupported batch type for DiffusionBridgeMethod: {type(batch)}"
        )

    def prepare_diffusion_inputs(
        self,
        parsed_batch: Any,
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(parsed_batch, dict):
            raise TypeError(
                f"parse_batch must return a dict, got {type(parsed_batch)}."
            )

        x0   = parsed_batch.get("x0",   None)
        cond = parsed_batch.get("cond", None)
        meta = parsed_batch.get("meta", None)

        if x0 is None:
            raise KeyError("DiffusionBridgeMethod requires 'x0' (target) in parsed batch.")
        if cond is None:
            raise KeyError("DiffusionBridgeMethod requires 'cond' (source) in parsed batch.")
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a torch.Tensor.")
        if not torch.is_tensor(cond):
            raise TypeError("cond (source) must be a torch.Tensor.")

        return {"x0": x0, "cond": cond, "meta": meta}

    # ------------------------------------------------------------------
    # core training state
    # ------------------------------------------------------------------
    def predict_x0_recursive(
        self,
        xt: torch.Tensor,
        source: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recursive x0 estimation during training:

            x0_r^{k+1} = model(cat(x_t, source), t, x_r=x0_r^k)

        Runs ``self.n_recursions`` iterations with optional early stopping.
        """
        x0_r = torch.zeros_like(xt)
        for _ in range(self.n_recursions):
            x0_rp1 = self.model(torch.cat((xt, source), dim=1), t, x_r=x0_r)

            if self.consistency_threshold > 0.0:
                change = torch.abs(x0_rp1 - x0_r).mean(dim=0).max()
                x0_r = x0_rp1
                if change.item() < self.consistency_threshold:
                    break
            else:
                x0_r = x0_rp1

        return x0_r

    def build_diffusion_state(
        self,
        inputs: Dict[str, Any],
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        x0     = inputs["x0"]
        source = inputs["cond"]
        meta   = inputs.get("meta", None)

        batch_size, device = self.batch_size_and_device_from_tensor(x0)

        t = self.sample_time(
            batch_size=batch_size,
            device=device,
            stage=stage,
            state=None,
        )

        process_state = self.build_process_state(
            x0=x0,
            t=t,
            cond=source,
        )

        state: Dict[str, Any] = {}
        state.update(inputs)
        state.update(process_state)
        state.setdefault("x0",   x0)
        state.setdefault("cond", source)
        state.setdefault("meta", meta)
        state.setdefault("t",    t)

        xt = state.get("xt", None)
        if xt is None:
            raise KeyError(
                "process.forward_state must provide 'xt' for DiffusionBridgeMethod."
            )

        x0_pred = self.predict_x0_recursive(xt, source, t)

        # Populate keys expected by loss terms
        state["x0_pred"] = x0_pred
        state["pred"]    = x0_pred
        state["target"]  = x0

        # Artifacts
        self.attach_artifact(state, "x0",     x0)
        self.attach_artifact(state, "source", source)
        self.attach_artifact(state, "xt",     xt)
        self.attach_artifact(state, "x0_pred", x0_pred)
        self.attach_artifact(state, "t",      t)

        # Optional sample at val/test
        sampled = self._maybe_sample(stage=stage, source=source)
        if sampled is not None:
            self.attach_artifact(state, f"{stage}_sample", sampled)

        return state

    # ------------------------------------------------------------------
    # inference / sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer(
        self,
        source: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Full reverse sampling: x_0 ~ p(x_0 | source).

        Delegates to ``process.sample_x0``.
        """
        src = source if source is not None else cond
        if src is None:
            raise ValueError(
                "DiffusionBridgeMethod.infer requires 'source' or 'cond'."
            )
        if self.process is None:
            raise ValueError("DiffusionBridgeMethod requires a process.")
        if not hasattr(self.process, "sample_x0"):
            raise AttributeError(
                "process must implement sample_x0(...) for DiffusionBridgeMethod inference."
            )
        return self.process.sample_x0(source=src, model=self.model, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        source: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.infer(source=source, cond=cond, **kwargs)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _maybe_sample(
        self,
        stage: str,
        source: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if stage not in {"val", "test"}:
            return None
        want = (stage == "val" and self.sample_on_val) or (stage == "test" and self.sample_on_test)
        if not want:
            return None
        try:
            return self.infer(source=source)
        except NotImplementedError:
            return None
        except Exception as e:
            logger.warning(
                f"Bridge sampling during '{stage}' failed and was skipped. "
                f"{type(e).__name__}: {e}"
            )
            return None
