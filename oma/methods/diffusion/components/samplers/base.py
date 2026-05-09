from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch


class BaseDiffusionSampler(ABC):
    """
    Base contract for diffusion samplers in OpenMedAxis.

    A sampler owns the iterative reverse / generation procedure used at
    inference time. It is intentionally separate from:

    - the diffusion method      -> training / evaluation pipeline skeleton
    - the diffusion process     -> forward path / x_t construction math
    - the diffusion objective   -> semantic meaning of model prediction

    Design goals
    ------------
    - Keep sampler swappable without changing method code.
    - Support deterministic and stochastic samplers.
    - Support future families beyond classic DDPM/DDIM.
    - Keep the API researcher-friendly and hackable.

    Typical usage
    -------------
    method.infer(...) -> sampler.sample(...)

    A concrete sampler may:
    - initialize x_T / starting state
    - build a reverse time schedule
    - repeatedly call model/process/objective helpers
    - return final sample and optionally intermediate artifacts
    """

    @abstractmethod
    def sample(
        self,
        *,
        model: Any,
        process: Any,
        cond: Optional[Any] = None,
        shape: Optional[torch.Size] = None,
        x_init: Optional[torch.Tensor] = None,
        method: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the full sampling / inference procedure.

        Parameters
        ----------
        model:
            The underlying network, usually called as model(x, t, cond=...).
        process:
            Diffusion/path process component.
        cond:
            Optional conditioning object.
        shape:
            Target tensor shape when a new initial noise/sample must be created.
        x_init:
            Optional explicit starting tensor. If given, sampler should prefer
            this over creating a new initial sample.
        method:
            Optional owning diffusion method. Useful for calling helper methods
            such as forward_model(...) or reconstruct_clean(...).

        Returns
        -------
        Any
            Usually the final generated sample tensor, but richer outputs are
            allowed if the method chooses to support them.
        """
        raise NotImplementedError

    def prepare_initial_state(
        self,
        *,
        shape: Optional[torch.Size],
        x_init: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Default initialization helper.

        Priority:
            1. use x_init if provided
            2. sample standard Gaussian noise with given shape
        """
        if x_init is not None:
            if not torch.is_tensor(x_init):
                raise TypeError("x_init must be a torch.Tensor when provided.")
            return x_init

        if shape is None:
            raise ValueError("Either `x_init` or `shape` must be provided.")

        if device is None:
            device = torch.device("cpu")

        if dtype is None:
            dtype = torch.float32

        return torch.randn(shape, device=device, dtype=dtype)

    def get_model_device_and_dtype(self, model: Any) -> Tuple[torch.device, torch.dtype]:
        """
        Infer device/dtype from the model when possible.
        """
        if hasattr(model, "parameters"):
            try:
                param = next(model.parameters())
                return param.device, param.dtype
            except StopIteration:
                pass

        return torch.device("cpu"), torch.float32

    def make_time_tensor(
        self,
        *,
        t_value: int | float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        """
        Build a batch-shaped time tensor filled with a scalar time value.
        """
        return torch.full(
            (batch_size,),
            fill_value=t_value,
            device=device,
            dtype=dtype,
        )

    def call_model(
        self,
        *,
        model: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        method: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Canonical model invocation.

        If the owning method is provided and exposes forward_model(...),
        prefer that so sampler behavior stays aligned with method conventions.
        """
        if method is not None and hasattr(method, "forward_model"):
            return method.forward_model(x=x, t=t, cond=cond, **kwargs)

        if cond is None:
            return model(x, t, **kwargs)

        return model(x, t, cond=cond, **kwargs)

    def reconstruct_clean(
        self,
        *,
        model_pred: Any,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        process: Optional[Any] = None,
        method: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Try to reconstruct/predict clean sample from model output.

        Priority:
            1. method.reconstruct_clean(...)
            2. process.predict_x0(...)
            3. raw model_pred
        """
        if state is None:
            state = {"xt": x, "t": t, "cond": cond}

        if method is not None and hasattr(method, "reconstruct_clean"):
            out = method.reconstruct_clean(model_pred, state)
            if out is not None:
                return out

        if process is not None and hasattr(process, "predict_x0"):
            out = process.predict_x0(
                model_pred=model_pred,
                xt=x,
                t=t,
                cond=cond,
                state=state,
            )
            if out is not None:
                return out

        return model_pred


class SingleStepSampler(BaseDiffusionSampler):
    """
    Minimal sampler useful for smoke tests and non-iterative inference.

    It performs one model call at a chosen time value and returns either:
    - reconstructed clean sample (preferred), or
    - raw model prediction
    """

    def __init__(
        self,
        t_value: int | float = 0,
        time_dtype: torch.dtype = torch.long,
    ) -> None:
        self.t_value = t_value
        self.time_dtype = time_dtype

    @torch.no_grad()
    def sample(
        self,
        *,
        model: Any,
        process: Any,
        cond: Optional[Any] = None,
        shape: Optional[torch.Size] = None,
        x_init: Optional[torch.Tensor] = None,
        method: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        device, dtype = self.get_model_device_and_dtype(model)
        x = self.prepare_initial_state(
            shape=shape,
            x_init=x_init,
            device=device,
            dtype=dtype,
        )

        if not torch.is_tensor(x):
            raise TypeError("Sampler initial state must be a torch.Tensor.")

        batch_size = int(x.shape[0])
        t = self.make_time_tensor(
            t_value=self.t_value,
            batch_size=batch_size,
            device=x.device,
            dtype=self.time_dtype,
        )

        model_pred = self.call_model(
            model=model,
            x=x,
            t=t,
            cond=cond,
            method=method,
            **kwargs,
        )

        return self.reconstruct_clean(
            model_pred=model_pred,
            x=x,
            t=t,
            cond=cond,
            process=process,
            method=method,
        )