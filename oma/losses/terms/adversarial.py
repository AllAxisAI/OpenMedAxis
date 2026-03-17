from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import LossState, StatefulLossTerm


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    if global_step < threshold:
        return value
    return weight


class _BaseAdversarialLossTerm(StatefulLossTerm):
    def __init__(
        self,
        discriminator: nn.Module,
        real_key: str = "target",
        fake_key: str = "pred",
        cond_key: Optional[str] = None,
        conditional: bool = False,
        disc_start: int = 0,
        weight: float = 1.0,
        name: Optional[str] = None,
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.discriminator = discriminator
        self.real_key = real_key
        self.fake_key = fake_key
        self.cond_key = cond_key
        self.conditional = conditional
        self.disc_start = int(disc_start)

    def validate(self, state: LossState) -> None:
        if self.fake_key not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key '{self.fake_key}' in loss state."
            )

        fake = state[self.fake_key]
        if not torch.is_tensor(fake):
            raise TypeError(
                f"State key '{self.fake_key}' must be a torch.Tensor, got {type(fake)}"
            )

        if self.real_key is not None:
            if self.real_key not in state:
                raise KeyError(
                    f"{self.__class__.__name__} expected key '{self.real_key}' in loss state."
                )
            real = state[self.real_key]
            if not torch.is_tensor(real):
                raise TypeError(
                    f"State key '{self.real_key}' must be a torch.Tensor, got {type(real)}"
                )

        if self.conditional:
            if self.cond_key is None:
                raise ValueError(
                    f"{self.__class__.__name__} received conditional=True but cond_key is None."
                )
            if self.cond_key not in state:
                raise KeyError(
                    f"{self.__class__.__name__} expected conditional key '{self.cond_key}' in loss state."
                )
            cond = state[self.cond_key]
            if not torch.is_tensor(cond):
                raise TypeError(
                    f"State key '{self.cond_key}' must be a torch.Tensor, got {type(cond)}"
                )

        if "global_step" not in state:
            raise KeyError(
                f"{self.__class__.__name__} expected key 'global_step' in loss state."
            )

    def _disc_input(self, x: torch.Tensor, state: LossState) -> torch.Tensor:
        if not self.conditional:
            return x.contiguous()

        cond = state[self.cond_key]
        return torch.cat((x.contiguous(), cond), dim=1)


class GeneratorAdversarialTerm(_BaseAdversarialLossTerm):
    """
    Generator-side adversarial loss.

    Belongs to group='main' by default.

    Computes:
        -mean(D(fake))

    Supports delayed activation using disc_start.
    """

    def __init__(
        self,
        discriminator: nn.Module,
        fake_key: str = "pred",
        cond_key: Optional[str] = None,
        conditional: bool = False,
        disc_start: int = 0,
        weight: float = 1.0,
        name: str = "g_adv",
        group: str = "main",
        state_logits_key: str = "logits_fake",
    ) -> None:
        super().__init__(
            discriminator=discriminator,
            real_key=None,
            fake_key=fake_key,
            cond_key=cond_key,
            conditional=conditional,
            disc_start=disc_start,
            weight=weight,
            name=name,
            group=group,
        )
        self.state_logits_key = state_logits_key

    def compute(self, state: LossState) -> torch.Tensor:
        fake = state[self.fake_key]
        global_step = int(state["global_step"])
        disc_factor = adopt_weight(self.weight, global_step, threshold=self.disc_start)

        if disc_factor == 0.0:
            return fake.new_zeros(())

        logits_fake = self.discriminator(self._disc_input(fake, state))
        state[self.state_logits_key] = logits_fake
        state[f"{self.name}_disc_factor"] = fake.new_tensor(disc_factor)

        g_loss = -torch.mean(logits_fake)
        return disc_factor * g_loss

    def build_logs(
        self,
        raw_value: torch.Tensor,
        weighted_value: torch.Tensor,
        state: LossState,
    ):
        split = state.get("split", "train")
        logs = {
            f"{split}/{self.name}": raw_value.detach(),
            f"{split}/{self.name}_weighted": weighted_value.detach(),
        }

        if self.state_logits_key in state:
            logs[f"{split}/{self.state_logits_key}"] = state[self.state_logits_key].detach().mean()

        disc_factor_key = f"{self.name}_disc_factor"
        if disc_factor_key in state:
            logs[f"{split}/{self.name}_disc_factor"] = state[disc_factor_key].detach()

        return logs


class DiscriminatorAdversarialTerm(_BaseAdversarialLossTerm):
    """
    Discriminator-side adversarial loss.

    Belongs to group='disc' by default.

    Supports:
        - hinge loss
        - vanilla logistic-style loss
        - delayed activation using disc_start
    """

    def __init__(
        self,
        discriminator: nn.Module,
        real_key: str = "target",
        fake_key: str = "pred",
        cond_key: Optional[str] = None,
        conditional: bool = False,
        disc_start: int = 0,
        weight: float = 1.0,
        name: str = "d_adv",
        group: str = "disc",
        mode: str = "hinge",
        state_real_logits_key: str = "logits_real",
        state_fake_logits_key: str = "logits_fake_detached",
    ) -> None:
        super().__init__(
            discriminator=discriminator,
            real_key=real_key,
            fake_key=fake_key,
            cond_key=cond_key,
            conditional=conditional,
            disc_start=disc_start,
            weight=weight,
            name=name,
            group=group,
        )

        if mode not in ("hinge", "vanilla"):
            raise ValueError(f"Unsupported adversarial mode '{mode}'. Use 'hinge' or 'vanilla'.")

        self.mode = mode
        self.state_real_logits_key = state_real_logits_key
        self.state_fake_logits_key = state_fake_logits_key

    def compute(self, state: LossState) -> torch.Tensor:
        real = state[self.real_key].detach()
        fake = state[self.fake_key].detach()
        global_step = int(state["global_step"])
        disc_factor = adopt_weight(self.weight, global_step, threshold=self.disc_start)

        if disc_factor == 0.0:
            return real.new_zeros(())

        logits_real = self.discriminator(self._disc_input(real, state))
        logits_fake = self.discriminator(self._disc_input(fake, state))

        state[self.state_real_logits_key] = logits_real
        state[self.state_fake_logits_key] = logits_fake
        state[f"{self.name}_disc_factor"] = real.new_tensor(disc_factor)

        if self.mode == "hinge":
            d_loss = hinge_d_loss(logits_real, logits_fake)
        else:
            d_loss = vanilla_d_loss(logits_real, logits_fake)

        return disc_factor * d_loss

    def build_logs(
        self,
        raw_value: torch.Tensor,
        weighted_value: torch.Tensor,
        state: LossState,
    ):
        split = state.get("split", "train")
        logs = {
            f"{split}/{self.name}": raw_value.detach(),
            f"{split}/{self.name}_weighted": weighted_value.detach(),
        }

        if self.state_real_logits_key in state:
            logs[f"{split}/{self.state_real_logits_key}"] = (
                state[self.state_real_logits_key].detach().mean()
            )
        if self.state_fake_logits_key in state:
            logs[f"{split}/{self.state_fake_logits_key}"] = (
                state[self.state_fake_logits_key].detach().mean()
            )

        disc_factor_key = f"{self.name}_disc_factor"
        if disc_factor_key in state:
            logs[f"{split}/{self.name}_disc_factor"] = state[disc_factor_key].detach()

        return logs


class FeatureMatchingLossTerm(StatefulLossTerm):
    """
    Optional feature-matching loss scaffold.

    This assumes the discriminator can return intermediate features when called
    with `return_features=True`.

    Not all discriminators support this, but this abstraction is useful for
    pix2pixHD-like setups later.
    """

    def __init__(
        self,
        discriminator: nn.Module,
        real_key: str = "target",
        fake_key: str = "pred",
        cond_key: Optional[str] = None,
        conditional: bool = False,
        weight: float = 1.0,
        name: str = "feature_matching",
        group: str = "main",
    ) -> None:
        super().__init__(weight=weight, name=name, group=group)
        self.discriminator = discriminator
        self.real_key = real_key
        self.fake_key = fake_key
        self.cond_key = cond_key
        self.conditional = conditional

    def validate(self, state: LossState) -> None:
        for key in [self.real_key, self.fake_key]:
            if key not in state:
                raise KeyError(f"{self.__class__.__name__} expected key '{key}' in loss state.")
            if not torch.is_tensor(state[key]):
                raise TypeError(f"State key '{key}' must be a torch.Tensor, got {type(state[key])}")

        if self.conditional:
            if self.cond_key is None:
                raise ValueError(
                    f"{self.__class__.__name__} received conditional=True but cond_key is None."
                )
            if self.cond_key not in state:
                raise KeyError(
                    f"{self.__class__.__name__} expected conditional key '{self.cond_key}' in loss state."
                )

    def _disc_input(self, x: torch.Tensor, state: LossState) -> torch.Tensor:
        if not self.conditional:
            return x.contiguous()
        cond = state[self.cond_key]
        return torch.cat((x.contiguous(), cond), dim=1)

    def compute(self, state: LossState) -> torch.Tensor:
        real = state[self.real_key].detach()
        fake = state[self.fake_key]

        try:
            _, real_feats = self.discriminator(self._disc_input(real, state), return_features=True)
            _, fake_feats = self.discriminator(self._disc_input(fake, state), return_features=True)
        except TypeError as e:
            raise TypeError(
                "Discriminator does not appear to support return_features=True, "
                "which is required for FeatureMatchingLossTerm."
            ) from e

        if len(real_feats) != len(fake_feats):
            raise ValueError(
                "FeatureMatchingLossTerm requires the same number of real and fake feature maps."
            )

        loss = 0.0
        for fr, ff in zip(real_feats, fake_feats):
            loss = loss + torch.mean(torch.abs(fr.detach() - ff))

        if isinstance(loss, float):
            fake = state[self.fake_key]
            loss = fake.new_zeros(())

        return loss