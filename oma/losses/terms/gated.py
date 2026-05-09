from __future__ import annotations

from typing import Optional

import torch

from ..base import LossOutput, LossState, LossTerm


class TimestepGatedLossTerm(LossTerm):
    """
    Wrapper that scales an inner loss term by how many batch samples
    satisfy a timestep condition.

    The gate is a soft scalar in [0, 1]:

        gate = mean(min_t <= t < max_t)

    The inner term runs at full strength when gate=1 (all samples qualify)
    and contributes nothing when gate=0 (no samples qualify). Between those
    extremes the loss is proportionally scaled — no hard cutoffs, no
    discontinuities.

    This is useful for:
    - Applying LPIPS only at low-noise timesteps (high quality x0_pred)
    - Delaying adversarial loss to low-t where the discriminator has signal
    - Any loss that is only meaningful within a specific timestep range

    Parameters
    ----------
    term:
        Any LossTerm to wrap.
    max_t:
        Upper bound (exclusive). Samples with t >= max_t do not contribute.
        ``None`` means no upper bound.
    min_t:
        Lower bound (inclusive). Samples with t < min_t do not contribute.
        ``None`` means no lower bound (default 0).
    t_key:
        State key for the timestep tensor. Default ``"t"``.
        If not found in state, the inner term runs without gating.

    Example: LPIPS only at low-noise timesteps
    -------------------------------------------
    ::

        import lpips
        lpips_net = lpips.LPIPS(net="alex")

        loss_fn = LossComposer([
            MSELossTerm(pred_key="eps_pred", target_key="eps_target", weight=1.0),
            TimestepGatedLossTerm(
                term=LPIPSLossTerm(
                    lpips_model=lpips_net,
                    pred_key="x0_pred",
                    target_key="x0",
                    weight=0.1,
                ),
                max_t=250,   # only when t < 250 (low noise, x0_pred is meaningful)
            ),
        ])

    Example: adversarial only in mid-range timesteps
    -------------------------------------------------
    ::

        TimestepGatedLossTerm(
            term=GeneratorAdversarialTerm(...),
            min_t=10,
            max_t=300,
        )
    """

    def __init__(
        self,
        term: LossTerm,
        max_t: Optional[float] = None,
        min_t: Optional[float] = None,
        t_key: str = "t",
    ) -> None:
        if max_t is None and min_t is None:
            raise ValueError(
                "TimestepGatedLossTerm requires at least one of max_t or min_t."
            )
        if max_t is not None and min_t is not None and min_t >= max_t:
            raise ValueError(
                f"min_t must be < max_t, got min_t={min_t}, max_t={max_t}."
            )

        # weight=1.0 — we don't apply it ourselves, forward is fully overridden
        super().__init__(weight=1.0, name=f"gated_{term.name}", group=term.group)

        self.inner_term = term
        self.max_t = float(max_t) if max_t is not None else None
        self.min_t = float(min_t) if min_t is not None else None
        self.t_key = t_key

    def compute(self, state: LossState) -> torch.Tensor:
        # Never called — forward is fully overridden.
        raise NotImplementedError(
            "TimestepGatedLossTerm overrides forward() directly."
        )

    def forward(self, state: LossState) -> LossOutput:
        t = state.get(self.t_key, None)
        split = state.get("split", "train")

        # No t in state — delegate fully without gating
        if t is None or not torch.is_tensor(t):
            return self.inner_term(state)

        # Build the mask
        t_f = t.float()
        mask = torch.ones_like(t_f, dtype=torch.bool)
        if self.min_t is not None:
            mask = mask & (t_f >= self.min_t)
        if self.max_t is not None:
            mask = mask & (t_f < self.max_t)

        gate = mask.float().mean()

        # All samples outside the gate — return zero, log the gate
        if gate.item() == 0.0:
            zero = gate.detach()
            return LossOutput(
                value=zero,
                logs={
                    f"{split}/{self.name}": zero,
                    f"{split}/{self.name}_gate": zero,
                },
                group=self.group,
                raw_value=zero,
                name=self.name,
            )

        # Run inner term and scale by gate
        out = self.inner_term(state)
        gated_value = gate * out.value

        logs = {
            f"{split}/{self.name}": gated_value.detach(),
            f"{split}/{self.name}_gate": gate.detach(),
        }

        return LossOutput(
            value=gated_value,
            logs=logs,
            group=out.group,
            raw_value=out.raw_value,
            name=self.name,
        )
