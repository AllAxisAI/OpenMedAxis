from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .base import BaseDataset


class NumpyDataset(BaseDataset):
    """
    Folder-based .npy dataset compatible with prepared layouts such as:

        data_dir/
            T1/
                train/
                    IXI002_slice_000.npy
                    ...
                val/
                    ...
                test/
                    ...
            T2/
                train/
                    IXI002_slice_000.npy
                    ...
            subject_ids.yaml   # optional

    Supports paired source-target loading.
    Intended as a lightweight dataset for simple experiments and
    compatibility with older projects.

    Returns either:
        dict:
            {
                "source": ...,
                "target": ...,
                "index": i,
                "id": "...",
                "meta": {...}
            }

        or legacy tuple:
            (target, source, i)
    """

    def __init__(
        self,
        data_dir: str | Path,
        stage: str,
        source_modality: str,
        target_modality: str,
        image_size: int | None,
        norm: bool = True,
        padding: bool = True,
        return_dict: bool = True,
        subject_ids_filename: str = "subject_ids.yaml",
    ) -> None:
        super().__init__(
            image_size=image_size,
            norm=norm,
            padding=padding,
            return_dict=return_dict,
        )

        self.data_dir = Path(data_dir)
        self.stage = stage
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.subject_ids_filename = subject_ids_filename

        self.source_files = self._load_file_list(self.source_modality)
        self.target_files = self._load_file_list(self.target_modality)

        if len(self.source_files) == 0:
            raise ValueError(
                f"No source .npy files found for modality={self.source_modality}, "
                f"stage={self.stage}, under {self.data_dir}"
            )

        if len(self.target_files) == 0:
            raise ValueError(
                f"No target .npy files found for modality={self.target_modality}, "
                f"stage={self.stage}, under {self.data_dir}"
            )

        if len(self.source_files) != len(self.target_files):
            raise ValueError(
                f"Source/target file count mismatch: "
                f"{len(self.source_files)} vs {len(self.target_files)}"
            )

        self._validate_pairing()

        first_target = np.load(self.target_files[0])
        self.original_shape = tuple(first_target.shape[-2:])

        self.subject_ids = self._load_subject_ids(self.subject_ids_filename)

    def _load_file_list(self, modality: str) -> List[Path]:
        modality_dir = self.data_dir / modality / self.stage
        if not modality_dir.exists():
            raise FileNotFoundError(f"Missing directory: {modality_dir}")

        return sorted([p for p in modality_dir.iterdir() if p.suffix == ".npy"])
        # files = [p for p in modality_dir.iterdir() if p.suffix == ".npy"]
        # files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # return files

    def _load_subject_ids(self, filename: str) -> Optional[np.ndarray]:
        subject_ids_path = self.data_dir / filename
        if not subject_ids_path.exists():
            return None

        with open(subject_ids_path, "r") as f:
            loaded = yaml.load(f, Loader=yaml.FullLoader)

        return np.array(loaded) if loaded is not None else None

    def _validate_pairing(self) -> None:
        """
        Validate that source and target filenames align one-to-one.

        Example:
            T1/train/IXI002_slice_010.npy
            T2/train/IXI002_slice_010.npy
        """
        source_names = [p.name for p in self.source_files]
        target_names = [p.name for p in self.target_files]

        if source_names != target_names:
            mismatch_examples = []
            for s_name, t_name in zip(source_names, target_names):
                if s_name != t_name:
                    mismatch_examples.append((s_name, t_name))
                if len(mismatch_examples) >= 5:
                    break

            raise ValueError(
                "Source and target file names do not align. "
                f"First mismatches: {mismatch_examples}"
            )

    def __len__(self) -> int:
        return len(self.source_files)

    def _load_array(self, path: Path) -> np.ndarray:
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(
                f"NumpyDataset currently expects 2D arrays per file, "
                f"but got shape {array.shape} from {path}"
            )
        return array.astype(np.float32)

    def __getitem__(self, index: int) -> Any:
        source_path = self.source_files[index]
        target_path = self.target_files[index]

        source = self._load_array(source_path)
        target = self._load_array(target_path)

        source = self._prepare_2d(source)
        target = self._prepare_2d(target)

        sample_id = source_path.stem

        meta: Dict[str, Any] = {
            "stage": self.stage,
            "source_modality": self.source_modality,
            "target_modality": self.target_modality,
            "source_path": str(source_path),
            "target_path": str(target_path),
            "original_shape": self.original_shape,
        }

        if self.return_dict:
            return self._prepare_sample_dict(
                index=index,
                sample_id=sample_id,
                source=source,
                target=target,
                meta=meta,
            )

        return target, source, index