from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class EvaluatorOutput:
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def __call__(
        self,
        *,
        stage: str,
        outputs: Mapping[str, Any],
        output_dir: Optional[str] = None,
        step: Optional[int] = None,
    ) -> EvaluatorOutput:
        ...