from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

#
from oma.losses.terms import L1LossTerm, L2LossTerm, CharbonnierLossTerm, HuberLossTerm
from oma.losses.base import LossTerm


class BaseDiffusionObjective(ABC):
    """
    Base contract for diffusion objectives in OpenMedAxis.

    An objective defines:
        - what the model prediction represents
        - how the target tensor is derived from diffusion state
        - which keys are exposed to the OMA loss engine
        - optionally how to reconstruct/predict the clean sample

    Important
    ---------
    This class does NOT own the numeric loss computation by default.
    Actual losses should be declared through OMA's loss system
    (e.g. LossComposer, L1LossTerm, MSELossTerm, etc.).
    """

    @property
    @abstractmethod
    def pred_key(self) -> str:
        """
        State key under which model prediction should be stored.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def target_key(self) -> str:
        """
        State key under which objective target should be stored.
        """
        raise NotImplementedError

    @abstractmethod
    def build_target(self, state: Dict[str, Any]) -> Any:
        """
        Build the supervision target from diffusion state.
        """
        raise NotImplementedError

    def populate_state(
        self,
        state: Dict[str, Any],
        model_pred: Any,
    ) -> Dict[str, Any]:
        """
        Populate the state with prediction and target keys expected by the
        framework loss engine.
        """
        state[self.pred_key] = model_pred
        state[self.target_key] = self.build_target(state)
        return state

    def reconstruct_clean(
        self,
        model_pred: Any,
        state: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """
        Optional helper to reconstruct/predict the clean sample from model output.
        """
        return None

    def default_loss_term(
        self,
        weight: float = 1.0,
        name: Optional[str] = None,
        group: str = "main",
        criterion: str = "mse",
    ) -> LossTerm:
        """
        Build a pre-wired loss term using this objective's pred_key and target_key.

        Parameters
        ----------
        weight:
            Scalar multiplier applied to this term's loss value.
        name:
            Log name for this term. Defaults to ``"{pred_key}_{criterion}"``.
        group:
            Optimization group (e.g. ``"main"``).
        criterion:
            Loss function. One of ``"mse"`` (default), ``"l1"``,
            ``"charbonnier"``, ``"huber"``.

        Example
        -------
        ::

            objective = EpsilonObjective()
            loss_fn = LossComposer([objective.default_loss_term(weight=1.0)])

            # swap to velocity — wiring follows automatically
            objective = VelocityObjective()
            loss_fn = LossComposer([objective.default_loss_term(weight=1.0)])
        """
        _criterion_map = {
            "mse":         L2LossTerm,
            "l2":          L2LossTerm,
            "l1":          L1LossTerm,
            "charbonnier": CharbonnierLossTerm,
            "huber":       HuberLossTerm,
        }
        term_cls = _criterion_map.get(criterion.lower())
        if term_cls is None:
            raise ValueError(
                f"Unsupported criterion '{criterion}'. "
                f"Choose from {list(_criterion_map.keys())}."
            )
        return term_cls(
            pred_key=self.pred_key,
            target_key=self.target_key,
            weight=weight,
            name=name or f"{self.pred_key}_{criterion.lower()}",
            group=group,
        )


class EpsilonObjective(BaseDiffusionObjective):
    """
    Standard epsilon / noise prediction objective.
    """

    @property
    def pred_key(self) -> str:
        return "eps_pred"

    @property
    def target_key(self) -> str:
        return "eps_target"

    def build_target(self, state: Dict[str, Any]) -> torch.Tensor:
        noise = state.get("noise", None)
        if noise is None:
            raise KeyError("EpsilonObjective requires state['noise'].")
        if not torch.is_tensor(noise):
            raise TypeError("EpsilonObjective expects state['noise'] to be a tensor.")
        return noise


class X0Objective(BaseDiffusionObjective):
    """
    Clean sample prediction objective.
    """

    @property
    def pred_key(self) -> str:
        return "x0_pred"

    @property
    def target_key(self) -> str:
        return "x0_target"

    def build_target(self, state: Dict[str, Any]) -> torch.Tensor:
        x0 = state.get("x0", None)
        if x0 is None:
            raise KeyError("X0Objective requires state['x0'].")
        if not torch.is_tensor(x0):
            raise TypeError("X0Objective expects state['x0'] to be a tensor.")
        return x0

    def reconstruct_clean(
        self,
        model_pred: Any,
        state: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        if torch.is_tensor(model_pred):
            return model_pred
        return None


class ResidualObjective(BaseDiffusionObjective):
    """
    Residual prediction objective.

    By default:
        residual_target = x0 - xt

    but the reference tensor can be changed.
    """

    def __init__(self, reference_key: str = "xt") -> None:
        self.reference_key = reference_key

    @property
    def pred_key(self) -> str:
        return "residual_pred"

    @property
    def target_key(self) -> str:
        return "residual_target"

    def build_target(self, state: Dict[str, Any]) -> torch.Tensor:
        x0 = state.get("x0", None)
        ref = state.get(self.reference_key, None)

        if x0 is None:
            raise KeyError("ResidualObjective requires state['x0'].")
        if ref is None:
            raise KeyError(
                f"ResidualObjective requires state['{self.reference_key}']."
            )

        if not torch.is_tensor(x0) or not torch.is_tensor(ref):
            raise TypeError(
                "ResidualObjective expects tensor-valued state entries."
            )

        return x0 - ref


class VelocityObjective(BaseDiffusionObjective):
    """
    Placeholder-friendly generic velocity objective.

    The exact velocity target depends on the process/path parameterization.
    The process is expected to write a precomputed tensor into state.
    """

    def __init__(self, velocity_key: str = "velocity_target") -> None:
        self.velocity_key = velocity_key

    @property
    def pred_key(self) -> str:
        return "velocity_pred"

    @property
    def target_key(self) -> str:
        return "velocity_target"

    def build_target(self, state: Dict[str, Any]) -> torch.Tensor:
        target = state.get(self.velocity_key, None)
        if target is None:
            raise KeyError(
                f"VelocityObjective requires state['{self.velocity_key}']."
            )
        if not torch.is_tensor(target):
            raise TypeError(
                f"VelocityObjective expects state['{self.velocity_key}'] to be a tensor."
            )
        return target