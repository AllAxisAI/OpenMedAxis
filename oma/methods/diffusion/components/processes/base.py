from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseDiffusionProcess(ABC):
    """
    Base contract for diffusion / path processes in OpenMedAxis.

    A process defines how an intermediate state is constructed from a clean
    sample and a time variable.

    This abstraction is intentionally broader than classic Gaussian diffusion.
    It should also support:
        - bridge-style processes
        - latent diffusion processes
        - flow/interpolation paths
        - custom research variants

    Expected responsibilities
    -------------------------
    A concrete process may define:
        - how time/timestep is sampled
        - how x_t (or equivalent intermediate state) is constructed
        - optional auxiliaries needed by losses or samplers
        - optional clean reconstruction helper from model prediction

    Notes
    -----
    - `t` may be discrete or continuous.
    - returned dictionaries should be rich and explicit rather than minimal.
    - keys like `xt`, `noise`, and `process_aux` are encouraged where relevant.
    """

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        stage: str,
        state: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Optional time sampling hook.

        Concrete processes may override this if they want to own timestep/time
        sampling directly. If not overridden, BaseDiffusionMethod may use a
        dedicated time sampler component instead.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sample_time(...)."
        )

    @abstractmethod
    def forward_state(
        self,
        *,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Construct the process state at time `t`.

        Parameters
        ----------
        x0:
            Clean/reference sample.
        t:
            Time or timestep tensor.
        cond:
            Optional conditioning object.
        noise:
            Optional externally supplied stochastic variable.

        Returns
        -------
        dict
            Should typically include:
                - "xt": intermediate state
            and may include:
                - "noise": the actual sampled/used noise
                - "x0": original clean input
                - "t": time tensor
                - "process_aux": coefficients or other metadata
        """
        raise NotImplementedError

    def predict_x0(
        self,
        *,
        model_pred: Any,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Optional helper to reconstruct/predict clean sample x0 from model output.

        Not every process can or should define this.
        """
        return None


class IdentityProcess(BaseDiffusionProcess):
    """
    Minimal no-op process useful for testing pipeline plumbing.

    Produces:
        xt = x0
    """

    def forward_state(
        self,
        *,
        x0: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "x0": x0,
            "xt": x0,
            "t": t,
            "noise": noise,
            "cond": cond,
            "process_aux": {},
        }

    def predict_x0(
        self,
        *,
        model_pred: Any,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Optional[torch.Tensor]:
        if torch.is_tensor(model_pred):
            return model_pred
        return None