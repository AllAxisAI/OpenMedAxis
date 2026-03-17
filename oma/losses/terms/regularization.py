from __future__ import annotations

from typing import Optional

import torch

from ..base import LossState, LossTerm


class KLLossTerm(LossTerm):
    """
    KL divergence term for posterior distributions such as
    DiagonalGaussianDistribution used in AutoencoderKL.

    Expects the state to contain a posterior object with a `.kl()` method.
    """

    def __init__(
        self,
        posterior_key: str = "posterior",
        weight: float = 1.0,
        name: str = "kl",
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.posterior_key = posterior_key

    def validate(self, state: LossState) -> None:
        if self.posterior_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.posterior_key}' in loss state."
            )

        posterior = state[self.posterior_key]
        if not hasattr(posterior, "kl"):
            raise TypeError(
                f"State key '{self.posterior_key}' must provide a `.kl()` method, "
                f"got object of type {type(posterior)}"
            )

    def compute(self, state: LossState) -> torch.Tensor:
        posterior = state[self.posterior_key]
        kl = posterior.kl()

        if not torch.is_tensor(kl):
            raise TypeError(
                f"{self.__class__.__name__}: posterior.kl() must return a torch.Tensor, "
                f"got {type(kl)}"
            )

        return kl.mean()


class LatentL1LossTerm(LossTerm):
    """
    L1 penalty on latent tensor.
    Useful for encouraging sparsity or constraining latent magnitude.
    """

    def __init__(
        self,
        latent_key: str = "latent",
        weight: float = 1.0,
        name: str = "latent_l1",
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.latent_key = latent_key

    def validate(self, state: LossState) -> None:
        if self.latent_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.latent_key}' in loss state."
            )
        latent = state[self.latent_key]
        if not torch.is_tensor(latent):
            raise TypeError(
                f"State key '{self.latent_key}' must be a torch.Tensor, got {type(latent)}"
            )

    def compute(self, state: LossState) -> torch.Tensor:
        z = state[self.latent_key]
        return torch.abs(z).mean()


class LatentL2LossTerm(LossTerm):
    """
    L2 penalty on latent tensor.
    Useful as an extra latent regularizer when desired.
    """

    def __init__(
        self,
        latent_key: str = "latent",
        weight: float = 1.0,
        name: str = "latent_l2",
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.latent_key = latent_key

    def validate(self, state: LossState) -> None:
        if self.latent_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.latent_key}' in loss state."
            )
        latent = state[self.latent_key]
        if not torch.is_tensor(latent):
            raise TypeError(
                f"State key '{self.latent_key}' must be a torch.Tensor, got {type(latent)}"
            )

    def compute(self, state: LossState) -> torch.Tensor:
        z = state[self.latent_key]
        return (z ** 2).mean()


class LogVarRegularizationTerm(LossTerm):
    """
    Optional regularization term for a learned log-variance parameter.

    This can be useful in objectives that learn logvar explicitly and you want
    to keep it from drifting too aggressively.
    """

    def __init__(
        self,
        logvar_key: str = "logvar",
        weight: float = 1.0,
        p: int = 2,
        name: str = "logvar_reg",
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        if p not in (1, 2):
            raise ValueError(f"Unsupported p={p}. Only 1 or 2 are allowed.")
        self.logvar_key = logvar_key
        self.p = p

    def validate(self, state: LossState) -> None:
        if self.logvar_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.logvar_key}' in loss state."
            )
        logvar = state[self.logvar_key]
        if not torch.is_tensor(logvar):
            raise TypeError(
                f"State key '{self.logvar_key}' must be a torch.Tensor, got {type(logvar)}"
            )

    def compute(self, state: LossState) -> torch.Tensor:
        logvar = state[self.logvar_key]
        if self.p == 1:
            return torch.abs(logvar).mean()
        return (logvar ** 2).mean()