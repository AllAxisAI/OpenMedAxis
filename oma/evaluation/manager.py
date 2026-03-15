from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .evaluators.base import Evaluator


class EvaluatorManager:
    def __init__(self, evaluators: Dict[str, Evaluator] | None = None) -> None:
        self.evaluators = evaluators or {}

    def run(
        self,
        *,
        stage: str,
        outputs: Mapping[str, Any],
        output_dir: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for name, evaluator in self.evaluators.items():
            out = evaluator(
                stage=stage,
                outputs=outputs,
                output_dir=output_dir,
                step=step,
            )

            for k, v in out.metrics.items():
                results[f"{stage}/{name}/{k}"] = v

            for k, v in out.artifacts.items():
                results[f"{stage}/{name}/{k}"] = v

        return results