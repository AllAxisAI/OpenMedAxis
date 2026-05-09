from __future__ import annotations

import logging # not sure if we want to use this or just rely on users attaching their own loggers to the method, but it can be helpful for debugging and is lightweight enough to include by default.
from typing import Any, Dict, Optional

import torch

from .base import BaseDiffusionMethod

logger = logging.getLogger(__name__)


class GaussianDiffusionMethod(BaseDiffusionMethod):
    """
    Generic Gaussian diffusion method for OpenMedAxis.

    This method keeps the OMA training style while delegating:
    - x_t construction to `process`
    - prediction semantics to `objective`
    - iterative inference to `sampler`

    Losses are expected to be computed by OMA's loss engine
    (e.g. LossComposer and loss terms).
    """

    def __init__(
        self,
        *args: Any,
        predict_clean: bool = True,
        infer_with_sampler: bool = True,
        attach_xt_artifact: bool = True,
        attach_x0_artifact: bool = True,
        attach_pred_artifact: bool = True,
        attach_target_artifact: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.predict_clean = predict_clean
        self.infer_with_sampler = infer_with_sampler

        self.attach_xt_artifact = attach_xt_artifact
        self.attach_x0_artifact = attach_x0_artifact
        self.attach_pred_artifact = attach_pred_artifact
        self.attach_target_artifact = attach_target_artifact

    # ------------------------------------------------------------------
    # batch parsing / input preparation
    # ------------------------------------------------------------------
    def parse_batch(self, batch: Any) -> Dict[str, Any]:
        """
        Default Gaussian diffusion batch parsing.

        Supported conventions
        ---------------------
        1. dict batch:
            {
                "x0": ...,
                "cond": ...,
                "meta": ...,
            }

           Also accepts common aliases:
                x, input, target -> x0

        2. tuple/list batch:
            (x0,)
            (x0, cond)
            (x0, cond, meta)

        3. tensor batch:
            x0
        """
        if isinstance(batch, dict):
            x0 = (
                batch.get("x0", None)
                if "x0" in batch
                else batch.get("target", batch.get("input", batch.get("x", None)))
            )
            if x0 is None:
                raise KeyError(
                    "GaussianDiffusionMethod dict batch must contain one of "
                    "['x0', 'target', 'input', 'x']."
                )

            return {
                "x0": x0,
                "cond": batch.get("cond", None),
                "meta": batch.get("meta", None),
                "raw_batch": batch,
            }

        if isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError("Received empty tuple/list batch.")

            if len(batch) == 1:
                return {
                    "x0": batch[0],
                    "cond": None,
                    "meta": None,
                    "raw_batch": batch,
                }

            if len(batch) == 2:
                return {
                    "x0": batch[0],
                    "cond": batch[1],
                    "meta": None,
                    "raw_batch": batch,
                }

            return {
                "x0": batch[0],
                "cond": batch[1],
                "meta": batch[2],
                "raw_batch": batch,
            }

        if torch.is_tensor(batch):
            return {
                "x0": batch,
                "cond": None,
                "meta": None,
                "raw_batch": batch,
            }

        raise TypeError(
            "Unsupported batch type for GaussianDiffusionMethod. "
            f"Got: {type(batch)}"
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
                f"parse_batch(...) must return a dict, got {type(parsed_batch)}."
            )

        x0 = parsed_batch.get("x0", None)
        cond = parsed_batch.get("cond", None)
        meta = parsed_batch.get("meta", None)

        if x0 is None:
            raise KeyError("prepare_diffusion_inputs requires parsed_batch['x0'].")

        if not torch.is_tensor(x0):
            raise TypeError("parsed_batch['x0'] must be a torch.Tensor.")

        return {
            "x0": x0,
            "cond": cond,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # main diffusion state builder
    # ------------------------------------------------------------------
    def build_diffusion_state(
        self,
        inputs: Dict[str, Any],
        stage: str,
        batch_idx: int,
        group: Optional[str] = None,
    ) -> Dict[str, Any]:
        x0 = inputs["x0"]
        cond = inputs.get("cond", None)
        meta = inputs.get("meta", None)

        batch_size, device = self.batch_size_and_device_from_tensor(x0)

        t = self.sample_time(
            batch_size=batch_size,
            device=device,
            stage=stage,
            state=None,
        )

        noise = self.sample_noise(x0)

        process_state = self.build_process_state(
            x0=x0,
            t=t,
            cond=cond,
            noise=noise,
            stage=stage,
            batch_idx=batch_idx,
            group=group,
        )

        state: Dict[str, Any] = {}
        state.update(inputs)
        state.update(process_state)

        state.setdefault("x0", x0)
        state.setdefault("cond", cond)
        state.setdefault("meta", meta)
        state.setdefault("t", t)
        state.setdefault("noise", noise)

        xt = state.get("xt", None)
        if xt is None:
            raise KeyError(
                "process.forward_state(...) must provide 'xt' for GaussianDiffusionMethod."
            )
        if not torch.is_tensor(xt):
            raise TypeError("state['xt'] must be a torch.Tensor.")

        model_pred = self.forward_model(
            x=xt,
            t=t,
            cond=cond,
        )

        state = self.apply_objective(state, model_pred)

        x0_pred = None
        if self.predict_clean:
            x0_pred = self.reconstruct_clean(model_pred, state)
            if x0_pred is not None:
                state["x0_pred"] = x0_pred

        self.attach_standard_artifacts(
            state=state,
            model_pred=model_pred,
            x0_pred=x0_pred,
        )

        # self.attach_metric(state, f"{stage}/t_mean", t.float().mean().detach())

        sampled = self.maybe_build_sample_artifact(
            stage=stage,
            cond=cond,
            x_shape=x0.shape,
        )
        if sampled is not None:
            self.attach_artifact(state, f"{stage}_sample", sampled)

        return state

    # ------------------------------------------------------------------
    # artifact helpers
    # ------------------------------------------------------------------
    def attach_standard_artifacts(
        self,
        *,
        state: Dict[str, Any],
        model_pred: Any,
        x0_pred: Optional[Any] = None,
    ) -> None:
        if self.attach_x0_artifact:
            self.attach_artifact(state, "x0", state.get("x0", None))

        if self.attach_xt_artifact:
            self.attach_artifact(state, "xt", state.get("xt", None))

        if self.attach_pred_artifact:
            self.attach_artifact(state, "model_pred", model_pred)

        if self.attach_target_artifact and self.objective is not None:
            target_key = getattr(self.objective, "target_key", None)
            if isinstance(target_key, str) and target_key in state:
                self.attach_artifact(state, target_key, state[target_key])

        if x0_pred is not None:
            self.attach_artifact(state, "x0_pred", x0_pred)

        self.attach_artifact(state, "t", state.get("t", None))

    def maybe_build_sample_artifact(
        self,
        *,
        stage: str,
        cond: Optional[Any],
        x_shape: torch.Size,
    ) -> Optional[Any]:
        if stage not in {"val", "test"}:
            return None

        want_sample = (
            (stage == "val" and self.sample_on_val)
            or (stage == "test" and self.sample_on_test)
        )
        if not want_sample:
            return None

        try:
            return self.infer(
                cond=cond,
                shape=x_shape,
                x_init=None,
                stage=stage,
            )
        except NotImplementedError:
            # sampling not supported by this method/sampler — silently skip
            return None
        except Exception as e:
            logger.warning(
                f"Sampling during '{stage}' failed and was skipped. "
                f"{type(e).__name__}: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # inference / sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer(
        self,
        cond: Optional[Any] = None,
        shape: Optional[torch.Size] = None,
        x_init: Optional[torch.Tensor] = None,
        stage: str = "test",
        **kwargs: Any,
    ) -> Any:
        """
        Main inference entrypoint.

        Behavior
        --------
        1. If sampler exists and infer_with_sampler=True:
             delegate to sampler
        2. Else if x_init is given:
             do one-step model forward and reconstruct clean sample if possible
        3. Else:
             raise informative error
        """
        sampler = self.get_sampler_for_stage(stage)

        if self.infer_with_sampler and sampler is not None:
            if hasattr(sampler, "sample"):
                return sampler.sample(
                    model=self.model,
                    process=self.process,
                    cond=cond,
                    shape=shape,
                    x_init=x_init,
                    method=self,
                    **kwargs,
                )
            raise AttributeError("sampler must implement sample(...).")

        if x_init is not None:
            if not torch.is_tensor(x_init):
                raise TypeError("x_init must be a tensor when provided.")

            batch_size, device = self.batch_size_and_device_from_tensor(x_init)
            t = self.sample_time(
                batch_size=batch_size,
                device=device,
                stage=stage,
                state=None,
            )

            pred = self.forward_model(x=x_init, t=t, cond=cond)

            maybe_x0 = self.reconstruct_clean(
                pred,
                {
                    "xt": x_init,
                    "t": t,
                    "cond": cond,
                },
            )
            return pred if maybe_x0 is None else maybe_x0

        raise NotImplementedError(
            "GaussianDiffusion.infer(...) requires either a sampler or x_init."
        )

    @torch.no_grad()
    def sample(
        self,
        cond: Optional[Any] = None,
        shape: Optional[torch.Size] = None,
        x_init: Optional[torch.Tensor] = None,
        stage: str = "test",
        **kwargs: Any,
    ) -> Any:
        return self.infer(
            cond=cond,
            shape=shape,
            x_init=x_init,
            stage=stage,
            **kwargs,
        )