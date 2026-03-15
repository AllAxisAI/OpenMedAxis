from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json


class BaseRecipe(ABC):
    """
    Base class for dataset preparation recipes in OMA.

    A recipe is responsible for turning raw dataset files into
    OMA-ready prepared data + manifests.
    """

    name: str = "base"
    supports_download: bool = False

    def __init__(
        self,
        raw_root: str | Path,
        prepared_root: Optional[str | Path] = None,
        manifests_root: Optional[str | Path] = None,
    ) -> None:
        self.raw_root = Path(raw_root).expanduser().resolve()

        base_root = self.raw_root.parent
        self.prepared_root = (
            Path(prepared_root).expanduser().resolve()
            if prepared_root is not None
            else base_root / "prepared"
        )
        self.manifests_root = (
            Path(manifests_root).expanduser().resolve()
            if manifests_root is not None
            else base_root / "manifests"
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError

    def download(self, *args: Any, **kwargs: Any) -> None:
        """
        Download raw dataset files if supported by this recipe.

        Dataset-specific recipes may override this method.
        """
        raise NotImplementedError(
            f"Automatic download is not supported for recipe '{self.name}'. "
            f"Please check describe() for dataset acquisition instructions."
        )

    @abstractmethod
    def verify(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def discover(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def assign_splits(
        self,
        items: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Shared helpers
    # ---------------------------------------------------------------------

    def ensure_roots(self) -> None:
        self.prepared_root.mkdir(parents=True, exist_ok=True)
        self.manifests_root.mkdir(parents=True, exist_ok=True)

    def ensure_dir(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(
        self,
        obj: Any,
        path: str | Path,
        indent: int = 2,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)

        return path

    def load_json(self, path: str | Path) -> Any:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_config(
        self,
        config: Any,
        filename: str = "recipe_config.json",
    ) -> Path:
        serializable = self._to_serializable(config)
        return self.save_json(serializable, self.manifests_root / filename)

    def make_split_manifest_path(self, split: str) -> Path:
        return self.manifests_root / f"{split}.json"

    def write_split_manifests(
        self,
        split_entries: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Path]:
        written: dict[str, Path] = {}
        for split, entries in split_entries.items():
            path = self.make_split_manifest_path(split)
            self.save_json(entries, path)
            written[split] = path
        return written

    def to_relative_path(self, path: str | Path, root: str | Path) -> str:
        path = Path(path).resolve()
        root = Path(root).resolve()
        return str(path.relative_to(root))

    def summarize_split_sizes(
        self,
        split_items: dict[str, list[Any]],
    ) -> dict[str, int]:
        return {split: len(items) for split, items in split_items.items()}

    # ---------------------------------------------------------------------
    # Internal serialization helper
    # ---------------------------------------------------------------------

    def _to_serializable(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, dict):
            return {str(k): self._to_serializable(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]

        if hasattr(obj, "__dict__"):
            return {
                k: self._to_serializable(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }

        return obj