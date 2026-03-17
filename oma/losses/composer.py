from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .base import LogDict, LossOutput, LossState, LossTerm


class LossComposer(nn.Module):
    """
    Compose multiple loss terms and aggregate them by optimization group.

    Each term:
        - reads from the shared `state` dict
        - returns a `LossOutput`
        - contributes to one optimization group, e.g. 'main' or 'disc'

    The composer returns:
        {
            "losses": {
                "main": ...,
                "disc": ...,
            },
            "logs": {...},
            "term_outputs": {
                "l1": LossOutput(...),
                "kl": LossOutput(...),
                ...
            }
        }
    """

    def __init__(self, terms: Optional[Iterable[LossTerm]] = None) -> None:
        super().__init__()
        self.terms = nn.ModuleList(list(terms) if terms is not None else [])

    def __len__(self) -> int:
        return len(self.terms)

    def add_term(self, term: LossTerm) -> None:
        self.terms.append(term)

    def groups(self) -> List[str]:
        return sorted({term.group for term in self.terms})

    def has_group(self, group: str) -> bool:
        return any(term.group == group for term in self.terms)

    def terms_by_group(self, group: str) -> List[LossTerm]:
        return [term for term in self.terms if term.group == group]

    def forward(self, state: LossState) -> Dict[str, object]:
        if len(self.terms) == 0:
            raise ValueError("LossComposer has no terms.")

        split = state.get("split", "train")

        losses_by_group: Dict[str, torch.Tensor] = {}
        logs: LogDict = {}
        term_outputs: Dict[str, LossOutput] = {}

        for idx, term in enumerate(self.terms):
            out = term(state)

            term_name = out.name or f"term_{idx}"
            if term_name in term_outputs:
                raise ValueError(
                    f"Duplicate loss term name detected: '{term_name}'. "
                    "Each term should have a unique name."
                )

            term_outputs[term_name] = out

            if out.group not in losses_by_group:
                losses_by_group[out.group] = out.value
            else:
                losses_by_group[out.group] = losses_by_group[out.group] + out.value

            for k, v in out.logs.items():
                if k in logs:
                    raise ValueError(
                        f"Duplicate log key detected: '{k}'. "
                        f"Please give unique names to loss terms."
                    )
                logs[k] = v

        for group_name, group_loss in losses_by_group.items():
            logs[f"{split}/{group_name}_loss"] = group_loss.detach()

        return {
            "losses": losses_by_group,
            "logs": logs,
            "term_outputs": term_outputs,
        }