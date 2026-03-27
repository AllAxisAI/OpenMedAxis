from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Literal
import glob
import random
import tarfile
import zipfile
import warnings
import gdown

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .base import BaseRecipe


BraTSVersion = Literal["2015", "2021"]

DOWNLOAD_URLS = {
    "2021": "https://drive.google.com/file/d/1eOn7rD0TxFByDNwGnzwgjcdclbVpwpm8/view?usp=sharing",
    "2015": "https://drive.google.com/uc?export=download&id=1Q2yDKH7ayoTHEUU5AFQ8H1utBdoYS7f7",
}

ARCHIVE_NAMES = {
    "2021": "BraTS2021_training_data.tar",
    "2015": "BRATS2015_Training.zip",
}


@dataclass
class BraTSPrepareConfig:
    version: BraTSVersion = "2021"

    modalities: Sequence[str] = ("t1", "t2", "flair", "t1ce")

    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_mode: str = field(
        default="random",
        metadata={
            "help": "How subjects are split into train/val/test.",
            "choices": ["random", "ordered", "fixed_counts"],
        },
    )
    fixed_split_counts: Tuple[int | None, int | None, int | None] | None = None

    slice_axis: int = field(
        default=None,
        metadata={
            "help": "Axis along which slices are extracted. 2=axial for BraTS2021 NIfTI. "
                    "BraTS2015 typically uses axis 0 because SimpleITK returns (D,H,W).",
            "choices": [0, 1, 2],
        },
    )
    slice_range: Tuple[int, int] | None = field(
        default=(27, 127),
        metadata={"help": "Slice range [start, end). None means full volume."},
    )
    rotate_k: int = field(
        default=None,
        metadata={"help": "Number of 90-degree rotations via np.rot90. 0 means no rotation."},
    )
    normalize: str = field(
        default=None,
        metadata={
            "help": "Per-slice normalization method.",
            "choices": ["none", "minmax_per_slice", "legacy_brats"],
        },
    )
    export_format: str = "npy"
    seed: int = 42

    export_test_mask: bool | None = None
    mask_source_modality: str = "t1"
    mask_threshold: float = 0.1

    use_hgg: bool | None = None
    use_lgg: bool | None = None

    cleanup_raw: bool = False


class BraTSRecipe(BaseRecipe):
    """
    Unified BraTS recipe for BraTS2015 and BraTS2021.

    BraTS2021 raw layout:
        raw_root/
            BraTS2021_00000/
                BraTS2021_00000_t1.nii.gz
                BraTS2021_00000_t2.nii.gz
                BraTS2021_00000_flair.nii.gz
                BraTS2021_00000_t1ce.nii.gz
                BraTS2021_00000_seg.nii.gz

    BraTS2015 raw layout:
        raw_root/
            HGG/
                patient_x/...
            LGG/
                patient_y/...
    """

    name = "brats"
    supports_download = True

    def __init__(
        self,
        raw_root: str | Path,
        prepared_root: str | Path | None = None,
        manifests_root: str | Path | None = None,
    ) -> None:
        super().__init__(
            raw_root=raw_root,
            prepared_root=prepared_root,
            manifests_root=manifests_root,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": "BraTS",
            "auto_download": self.supports_download,
            "versions": ["2015", "2021"],
            "modalities": ["t1", "t2", "flair", "t1ce", "seg"],
            "expected_structure": {
                "2021": {
                    "raw_root": str(self.raw_root),
                    "example": {
                        "subject_dir": "BraTS2021_00000/",
                        "files": [
                            "BraTS2021_00000_t1.nii.gz",
                            "BraTS2021_00000_t2.nii.gz",
                            "BraTS2021_00000_flair.nii.gz",
                            "BraTS2021_00000_t1ce.nii.gz",
                            "BraTS2021_00000_seg.nii.gz",
                        ],
                    },
                },
                "2015": {
                    "raw_root": str(self.raw_root),
                    "example": {
                        "HGG": "HGG/patient_x/...mha",
                        "LGG": "LGG/patient_y/...mha",
                    },
                },
            },
        }

    def download(
        self,
        version: BraTSVersion,
        *,
        force: bool = False,
    ) -> None:
        
        if version not in DOWNLOAD_URLS:
            raise ValueError(f"Unsupported BraTS version: {version}")
        url = DOWNLOAD_URLS[version]
        archive_name = ARCHIVE_NAMES[version]


        self.raw_root.mkdir(parents=True, exist_ok=True)
        archive_path = self.raw_root / archive_name

        if archive_path.exists() and not force:
            return

        if archive_path.exists() and force:
            archive_path.unlink()

        self._download_file(url, archive_path)
        # self._extract_tar(archive_path, self.raw_root)
        self._extract_archive(archive_path, self.raw_root)

    def verify(self, version: BraTSVersion) -> None:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"raw_root does not exist: {self.raw_root}")

        if version == "2021":
            subject_dirs = [p for p in self.raw_root.iterdir() if p.is_dir()]
            if not subject_dirs:
                raise ValueError(f"No subject directories found under {self.raw_root}")

            found_valid = any(self._subject_dir_has_nifti_2021(p) for p in subject_dirs)
            if not found_valid:
                raise ValueError(f"No valid BraTS2021-style subject folders found in {self.raw_root}")
            return

        if version == "2015":
            if not (self.raw_root / "HGG").exists() and not (self.raw_root / "LGG").exists():
                raise ValueError(
                    f"Expected HGG/ or LGG/ directories under raw_root for BraTS2015: {self.raw_root}"
                )

            found_any = False
            for grade_dir in [self.raw_root / "HGG", self.raw_root / "LGG"]:
                if not grade_dir.exists():
                    continue
                for patient_dir in grade_dir.iterdir():
                    if patient_dir.is_dir() and self._find_all_mha_files(patient_dir):
                        found_any = True
                        break
                if found_any:
                    break

            if not found_any:
                raise ValueError(f"No .mha files found under BraTS2015 raw_root: {self.raw_root}")
            return

        raise ValueError(f"Unsupported BraTS version: {version}")

    def discover(self, version: BraTSVersion, config: BraTSPrepareConfig) -> List[Dict[str, Any]]:
        if version == "2021":
            return self._discover_2021()

        if version == "2015":
            return self._discover_2015(config)

        raise ValueError(f"Unsupported BraTS version: {version}")

    def assign_splits(
        self,
        items: List[Dict[str, Any]],
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        split_mode: str = "random",
        fixed_split_counts: Tuple[int | None, int | None, int | None] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        items = sorted(items, key=lambda x: x["subject_id"])

        if split_mode == "fixed_counts":
            if fixed_split_counts is None:
                raise ValueError("fixed_split_counts must be provided for split_mode='fixed_counts'")

            a, b, c = fixed_split_counts
            n = len(items)

            n_train = 0 if a is None else a
            n_val = 0 if b is None else b
            n_test = n - n_train - n_val if c is None else c

            if n_train + n_val + n_test > n:
                raise ValueError(
                    f"fixed_split_counts={fixed_split_counts} exceed dataset size {n}"
                )

            return {
                "train": items[:n_train],
                "val": items[n_train:n_train + n_val],
                "test": items[n_train + n_val:n_train + n_val + n_test],
            }

        if len(split_ratio) != 3:
            raise ValueError("split_ratio must have length 3")

        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

        if split_mode == "random":
            rng = random.Random(seed)
            items = list(items)
            rng.shuffle(items)
        elif split_mode != "ordered":
            raise ValueError(f"Unsupported split_mode: {split_mode}")

        n = len(items)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        return {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:],
        }

    def prepare(
        self,
        config: BraTSPrepareConfig | None = None,
        download: bool = False,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        config = config or BraTSPrepareConfig()
        config = self._apply_version_defaults(config)
        self._validate_config(config)
        print(f"Preparing BraTS with config: {asdict(config)}")

        if config.export_format.lower() != "npy":
            raise ValueError(
                f"Only export_format='npy' is supported in BraTSRecipe v1, got {config.export_format}"
            )

        if download:
          

            print(f"Downloading BraTS version {config.version}")
            self.download(
                version=config.version,
                force=force_download,
            )

        self.verify(config.version)
        self.ensure_roots()

        subjects = self.discover(config.version, config)
        filtered_subjects = self._filter_subjects_by_modalities(subjects, config.modalities)

        if not filtered_subjects:
            raise ValueError(
                f"No subjects found containing all requested modalities: {config.modalities}"
            )

        split_subjects = self.assign_splits(
            filtered_subjects,
            split_ratio=config.split_ratio,
            seed=config.seed,
            split_mode=config.split_mode,
            fixed_split_counts=config.fixed_split_counts,
        )

        split_entries: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for split_name, subject_items in split_subjects.items():
            # for subject in subject_items:
            for subject in tqdm(subject_items, desc=f"Processing {split_name} split"):
                entries = self._export_subject(
                    subject=subject,
                    split=split_name,
                    config=config,
                )
                split_entries[split_name].extend(entries)

        manifest_paths = self.write_split_manifests(split_entries)
        self.save_config(config)

        if config.cleanup_raw:
            print("Cleaning up raw files...")
            for item in tqdm(subjects, desc="Cleaning raw files"):
                for path in item["paths"].values():
                    try:
                        Path(path).unlink()
                    except Exception as e:
                        print(f"Warning: failed to delete {path}: {e}")
                        
            

        return {
            "recipe": self.name,
            "version": config.version,
            "raw_root": str(self.raw_root),
            "prepared_root": str(self.prepared_root),
            "manifests_root": str(self.manifests_root),
            "num_subjects": len(filtered_subjects),
            "split_subject_counts": self.summarize_split_sizes(split_subjects),
            "split_sample_counts": self.summarize_split_sizes(split_entries),
            "manifest_paths": {k: str(v) for k, v in manifest_paths.items()},
            "config": asdict(config),
        }

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _discover_2021(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        for subject_dir in sorted([p for p in self.raw_root.iterdir() if p.is_dir()]):
            subject_id = subject_dir.name
            paths: Dict[str, str] = {}

            for modality in ("t1", "t2", "flair", "t1ce", "seg"):
                f1 = subject_dir / f"{subject_id}_{modality}.nii.gz"
                f2 = subject_dir / f"{subject_id}_{modality}.nii"
                if f1.exists():
                    paths[modality] = str(f1)
                elif f2.exists():
                    paths[modality] = str(f2)

            if paths:
                entries.append({"subject_id": subject_id, "paths": paths})

        if not entries:
            raise ValueError("No BraTS2021 subject files discovered.")

        return entries

    def _discover_2015(self, config: BraTSPrepareConfig) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []

        grade_roots: List[Path] = []
        if config.use_hgg:
            grade_roots.append(self.raw_root / "HGG")
        if config.use_lgg:
            grade_roots.append(self.raw_root / "LGG")

        for grade_root in grade_roots:
            if not grade_root.exists():
                continue

            for patient_dir in sorted([p for p in grade_root.iterdir() if p.is_dir()]):
                subject_id = patient_dir.name
                paths: Dict[str, str] = {}

                for mha_path in self._find_all_mha_files(patient_dir):
                    modality = self._extract_2015_modality(mha_path.name)
                    if modality is not None and modality not in paths:
                        paths[modality] = str(mha_path)

                if paths:
                    entries.append({"subject_id": subject_id, "paths": paths})

        if not entries:
            raise ValueError("No BraTS2015 patient files discovered.")

        return entries

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_subject(
        self,
        subject: Dict[str, Any],
        split: str,
        config: BraTSPrepareConfig,
    ) -> List[Dict[str, Any]]:
        arrays: Dict[str, np.ndarray] = {}

        for modality, path in subject["paths"].items():
            if config.version == "2021":
                arrays[modality] = nib.load(path).get_fdata()
            elif config.version == "2015":
                img = sitk.ReadImage(path)
                arrays[modality] = sitk.GetArrayFromImage(img)
            else:
                raise ValueError(f"Unsupported BraTS version: {config.version}")

        shapes = {k: v.shape for k, v in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Shape mismatch for subject {subject['subject_id']}: {shapes}")

        shape = next(iter(shapes.values()))
        axis = config.slice_axis
        num_slices = shape[axis]
        start, end = self._resolve_slice_range(config.slice_range, num_slices)

        entries: List[Dict[str, Any]] = []

        for slice_idx in range(start, end):
            saved_paths: Dict[str, str] = {}

            for modality, volume in arrays.items():
                slice_data = self._extract_slice(volume, axis=axis, index=slice_idx)

                if config.rotate_k != 0:
                    slice_data = np.rot90(slice_data, k=config.rotate_k)

                if modality == "seg":
                    slice_data = slice_data.astype(np.int16)
                else:
                    slice_data = self._normalize_slice(slice_data, config.normalize)

                out_dir = self.prepared_root / modality / split
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"{subject['subject_id']}_slice_{slice_idx:03d}.npy"
                np.save(out_path, slice_data)

                saved_paths[modality] = self.to_relative_path(out_path, self.prepared_root)

            if split == "test" and config.export_test_mask:
                if config.mask_source_modality not in arrays:
                    raise ValueError(
                        f"mask_source_modality={config.mask_source_modality} not found in subject "
                        f"{subject['subject_id']}"
                    )

                mask_slice = self._extract_slice(
                    arrays[config.mask_source_modality],
                    axis=axis,
                    index=slice_idx,
                )
                if config.rotate_k != 0:
                    mask_slice = np.rot90(mask_slice, k=config.rotate_k)
                mask_slice = self._normalize_slice(mask_slice, config.normalize)
                mask = (mask_slice > config.mask_threshold).astype(np.uint8)

                mask_dir = self.prepared_root / "mask" / split
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{subject['subject_id']}_slice_{slice_idx:03d}.npy"
                np.save(mask_path, mask)
                saved_paths["mask"] = self.to_relative_path(mask_path, self.prepared_root)

            entries.append(
                {
                    "id": f"{subject['subject_id']}_slice_{slice_idx:03d}",
                    "subject_id": subject["subject_id"],
                    "paths": saved_paths,
                    "meta": {
                        "version": config.version,
                        "slice_index": slice_idx,
                        "slice_axis": axis,
                    },
                }
            )

        return entries

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _filter_subjects_by_modalities(
        self,
        subjects: List[Dict[str, Any]],
        modalities: Sequence[str],
    ) -> List[Dict[str, Any]]:
        required = {m.lower() for m in modalities}
        filtered: List[Dict[str, Any]] = []

        for subject in subjects:
            available = set(subject["paths"].keys())
            if required.issubset(available):
                filtered.append(
                    {
                        "subject_id": subject["subject_id"],
                        "paths": {m: subject["paths"][m] for m in subject["paths"] if m in required or m == "seg"},
                    }
                )

        return filtered

    def _subject_dir_has_nifti_2021(self, subject_dir: Path) -> bool:
        return any(subject_dir.glob("*.nii")) or any(subject_dir.glob("*.nii.gz"))

    def _find_all_mha_files(self, root_dir: Path) -> List[Path]:
        return sorted(root_dir.rglob("*.mha"))

    def _extract_2015_modality(self, filename: str) -> str | None:
        stem = Path(filename).stem
        if "T1c" in stem or "T1C" in stem:
            return "t1ce"
        if "T1" in stem:
            return "t1"
        if "T2" in stem:
            return "t2"
        if "FLAIR" in stem or "Flair" in stem:
            return "flair"
        return None

    def _resolve_slice_range(
        self,
        slice_range: Tuple[int, int] | None,
        num_slices: int,
    ) -> Tuple[int, int]:
        if slice_range is None:
            return 0, num_slices

        start, end = slice_range
        start = max(0, start)
        end = min(num_slices, end)

        if start >= end:
            raise ValueError(f"Invalid slice_range after clipping: ({start}, {end})")

        return start, end

    def _extract_slice(self, volume: np.ndarray, axis: int, index: int) -> np.ndarray:
        if axis == 0:
            return volume[index, :, :]
        if axis == 1:
            return volume[:, index, :]
        if axis == 2:
            return volume[:, :, index]
        raise ValueError(f"slice_axis must be one of 0, 1, 2, got {axis}")

    def _normalize_slice(self, slice_data: np.ndarray, method: str) -> np.ndarray:
        slice_data = slice_data.astype(np.float32)

        if method == "none":
            return slice_data

        if method == "legacy_brats":
            slice_data = slice_data - slice_data.min()
            return slice_data / (slice_data.max() + 1e-8)

        if method == "minmax_per_slice":
            mn = slice_data.min()
            mx = slice_data.max()
            return (slice_data - mn) / (mx - mn + 1e-8)

        raise ValueError(f"Unsupported normalize method: {method}")

    def _download_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)

        gdown.download(url, str(destination), quiet=False, fuzzy=True)


    def _extract_archive(self, archive_path: Path, extract_dir: Path) -> None:
        extract_dir.mkdir(parents=True, exist_ok=True)

        name = archive_path.name.lower()

        if name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz"):
            with tarfile.open(archive_path, "r:*") as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                    tar.extract(member, path=extract_dir)

        elif name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                members = z.namelist()
                for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                    z.extract(member, path=extract_dir)

        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
    def _apply_version_defaults(self, config: BraTSPrepareConfig) -> BraTSPrepareConfig:
        cfg = replace(config)

        if cfg.version == "2021":
            if cfg.slice_axis is None:
                cfg.slice_axis = 2
            if cfg.rotate_k is None:
                cfg.rotate_k = -1
            if cfg.normalize is None:
                cfg.normalize = "legacy_brats"
            if cfg.export_test_mask is None:
                cfg.export_test_mask = True
            if cfg.use_hgg is None:
                cfg.use_hgg = True
            if cfg.use_lgg is None:
                cfg.use_lgg = True

        elif cfg.version == "2015":
            if cfg.slice_axis is None:
                cfg.slice_axis = 0
            if cfg.rotate_k is None:
                cfg.rotate_k = 0
            if cfg.normalize is None:
                cfg.normalize = "legacy_brats"
            if cfg.export_test_mask is None:
                cfg.export_test_mask = True
            if cfg.use_hgg is None:
                cfg.use_hgg = False
            if cfg.use_lgg is None:
                cfg.use_lgg = True

        else:
            raise ValueError(f"Unsupported BraTS version: {cfg.version}")

        return cfg
    
    def _validate_config(self, config: BraTSPrepareConfig) -> None:
        if config.export_format != "npy":
            raise ValueError(f"Only export_format='npy' is supported, got {config.export_format}")

        if config.slice_axis not in {0, 1, 2}:
            raise ValueError(f"slice_axis must be 0, 1, or 2, got {config.slice_axis}")

        if config.normalize not in {"none", "minmax_per_slice", "legacy_brats"}:
            raise ValueError(
                f"normalize must be one of ['none', 'minmax_per_slice', 'legacy_brats'], got {config.normalize}"
            )

        if config.split_mode not in {"random", "ordered", "fixed_counts"}:
            raise ValueError(
                f"split_mode must be one of ['random', 'ordered', 'fixed_counts'], got {config.split_mode}"
            )

        if config.split_mode == "fixed_counts" and config.fixed_split_counts is None:
            raise ValueError("fixed_split_counts must be provided when split_mode='fixed_counts'")

        if config.slice_range is not None:
            start, end = config.slice_range
            if start < 0 or end <= start:
                raise ValueError(f"Invalid slice_range: {config.slice_range}")

        if config.version == "2015":
            if not (config.use_hgg or config.use_lgg):
                raise ValueError("For BraTS2015, at least one of use_hgg or use_lgg must be True")

            if config.slice_axis != 0:
                warnings.warn(
                    "BraTS2015 usually uses slice_axis=0 because SimpleITK returns (D,H,W).",
                    stacklevel=2,
                )

            if config.rotate_k != 0:
                warnings.warn(
                    "BraTS2015 usually uses rotate_k=0 in the original preprocessing.",
                    stacklevel=2,
                )

        if config.version == "2021":
            if config.slice_axis != 2:
                warnings.warn(
                    "BraTS2021 usually uses slice_axis=2 for axial slicing.",
                    stacklevel=2,
                )

            if config.use_hgg is not True or config.use_lgg is not True:
                warnings.warn(
                    "use_hgg/use_lgg are not relevant for BraTS2021 and will be ignored.",
                    stacklevel=2,
                )