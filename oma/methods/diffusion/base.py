from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Tuple

import torch

from oma.methods.base import GroupedLossMethod


class BaseDiffusionMethod(GroupedLossMethod, ABC):
    """
    Stable diffusion-facing base class for OpenMedAxis.

    Design goals
    ------------
    - Preserve OMA's method-first style.
    - Use GroupedLossMethod as the engine for complex and future multi-group setups.
    - Keep diffusion experimentation localized to overridable hooks and swappable
      components such as:
          * process
          * objective
          * sampler
          * time_sampler
    - Keep loss computation in OMA's loss engine rather than hardcoding it into
      the objective layer.

    Typical flow
    ------------
    1. parse_batch(...)
    2. prepare_diffusion_inputs(...)
    3. build diffusion state (t, noise, xt, ...)
    4. forward model
    5. objective populates prediction/target keys into state
    6. GroupedLossMethod / LossComposer computes actual losses
    """

    def __init__(
        self,
        *args: Any,
        process: Optional[Any] = None,
        objective: Optional[Any] = None,
        sampler: Optional[Any] = None,
        time_sampler: Optional[Any] = None,
        val_sampler: Optional[Any] = None,
        test_sampler: Optional[Any] = None,
        sample_on_val: bool = False,
        sample_on_test: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.process = process
        self.objective = objective
        self.sampler = sampler
        self.time_sampler = time_sampler

        self.val_sampler = val_sampler
        self.test_sampler = test_sampler

        self.sample_on_val = sample_on_val
        self.sample_on_test = sample_on_test

    # ------------------------------------------------------------------
    # researcher-facing hooks
    # ------------------------------------------------------------------
    def parse_batch(self, batch: Any) -> Any:
        """
        Parse raw batch into a diffusion-friendly structure.

        Default: pass-through.
        """
        return batch

    def prepare_diffusion_inputs(
        self,
        parsed_batch: Any,
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare canonical diffusion inputs before process/model/objective logic.

        Expected returned dict may include keys like:
            - x0
            - cond
            - source
            - target
            - meta
        """
        raise NotImplementedError

    def build_diffusion_state(
        self,
        inputs: Dict[str, Any],
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main diffusion state construction hook.

        Child methods usually:
            - sample time
            - sample noise
            - construct xt with process
            - call model
            - let objective populate state keys
            - optionally reconstruct clean sample
            - attach artifacts / method metrics

        Must return a dict suitable for OMA's loss engine.
        """
        raise NotImplementedError

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self.infer(*args, **kwargs)

    # ------------------------------------------------------------------
    # bridge to GroupedLossMethod
    # ------------------------------------------------------------------
    def build_state(
        self,
        batch: Any,
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stable OMA-facing state builder.

        This remains thin on purpose.
        """
        parsed = self.parse_batch(batch)

        inputs = self.prepare_diffusion_inputs(
            parsed_batch=parsed,
            stage=stage,
            batch_idx=batch_idx,
            group=group,
        )
        if not isinstance(inputs, dict):
            raise TypeError(
                f"prepare_diffusion_inputs(...) must return a dict, got {type(inputs)}."
            )

        state = self.build_diffusion_state(
            inputs=inputs,
            stage=stage,
            batch_idx=batch_idx,
            group=group,
        )
        if not isinstance(state, dict):
            raise TypeError(
                f"build_diffusion_state(...) must return a dict, got {type(state)}."
            )

        state = dict(state)
        state.setdefault("batch", batch)
        state.setdefault("parsed_batch", parsed)
        state.setdefault("inputs", inputs)
        state.setdefault("split", stage)
        state.setdefault("batch_idx", batch_idx)
        state.setdefault("group", group)
        state.setdefault("global_step", int(self.global_step))
        state.setdefault("_artifacts", {})
        state.setdefault("_method_metrics", {})

        return state

    # ------------------------------------------------------------------
    # reusable helpers
    # ------------------------------------------------------------------
    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        *,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Sample time/timestep values.

        Priority:
            1. time_sampler.sample(...)
            2. process.sample_time(...)
        """
        if self.time_sampler is not None:
            if not hasattr(self.time_sampler, "sample"):
                raise AttributeError("time_sampler must implement sample(...).")
            return self.time_sampler.sample(
                batch_size=batch_size,
                device=device,
                stage=stage,
                state=state,
            )

        if self.process is not None and hasattr(self.process, "sample_time"):
            return self.process.sample_time(
                batch_size=batch_size,
                device=device,
                stage=stage,
                state=state,
            )

        raise NotImplementedError(
            "No time sampling mechanism found. "
            "Provide `time_sampler` or implement `process.sample_time(...)`."
        )

    def sample_noise(self, like: torch.Tensor) -> torch.Tensor:
        """
        Default Gaussian noise helper.
        """
        return torch.randn_like(like)

    def forward_model(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Stable model calling convention across diffusion families.
        """
        if self.model is None:
            raise ValueError("BaseDiffusionMethod requires `self.model` to exist.")

        if cond is None:
            return self.model(x, t, **kwargs)
        return self.model(x, t, cond=cond, **kwargs)

    def build_process_state(
        self,
        *,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Delegate x_t or path construction to the process component.
        """
        if self.process is None:
            raise ValueError("BaseDiffusionMethod requires `self.process`.")

        if not hasattr(self.process, "forward_state"):
            raise AttributeError("process must implement forward_state(...).")

        out = self.process.forward_state(
            x0=x0,
            t=t,
            cond=cond,
            noise=noise,
            **kwargs,
        )
        if not isinstance(out, dict):
            raise TypeError(
                f"process.forward_state(...) must return a dict, got {type(out)}."
            )
        return out

    def apply_objective(
        self,
        state: Dict[str, Any],
        model_pred: Any,
    ) -> Dict[str, Any]:
        """
        Let the objective populate OMA loss-engine keys into state.

        At minimum this usually writes:
            - objective-specific pred key
            - objective-specific target key

        This method also stores a generic 'model_pred' key for convenience.
        """
        state["model_pred"] = model_pred

        if self.objective is None:
            return state

        if not hasattr(self.objective, "populate_state"):
            raise AttributeError("objective must implement populate_state(...).")

        out = self.objective.populate_state(state, model_pred)
        if out is None:
            return state
        if not isinstance(out, dict):
            raise TypeError(
                f"objective.populate_state(...) must return a dict or None, got {type(out)}."
            )
        return out

    def reconstruct_clean(
        self,
        model_pred: Any,
        state: Dict[str, Any],
    ) -> Any:
        """
        Try to reconstruct/predict clean sample from model output.

        Priority:
            1. objective.reconstruct_clean(...)
            2. process.predict_x0(...)
        """
        if self.objective is not None and hasattr(self.objective, "reconstruct_clean"):
            x0_pred = self.objective.reconstruct_clean(model_pred, state)
            if x0_pred is not None:
                return x0_pred

        if self.process is not None and hasattr(self.process, "predict_x0"):
            xt = state.get("xt", None)
            t = state.get("t", None)
            cond = state.get("cond", None)

            if xt is not None and t is not None:
                return self.process.predict_x0(
                    model_pred=model_pred,
                    xt=xt,
                    t=t,
                    cond=cond,
                    state=state,
                )

        return None

    def get_sampler_for_stage(self, stage: str) -> Optional[Any]:
        if stage == "val" and self.val_sampler is not None:
            return self.val_sampler
        if stage == "test" and self.test_sampler is not None:
            return self.test_sampler
        return self.sampler

    def batch_size_and_device_from_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[int, torch.device]:
        if not torch.is_tensor(tensor):
            raise TypeError(
                "batch_size_and_device_from_tensor expects a torch.Tensor input."
            )
        if tensor.ndim == 0:
            raise ValueError("Expected tensor with batch dimension, got scalar tensor.")
        return int(tensor.shape[0]), tensor.device

    def attach_artifact(
        self,
        state: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        artifacts = state.setdefault("_artifacts", {})
        artifacts[key] = value

    def attach_metric(
        self,
        state: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        metrics = state.setdefault("_method_metrics", {})
        metrics[key] = value