from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import random
import shutil
import tarfile
import tempfile
import urllib.request
from tqdm import tqdm

import nibabel as nib
import numpy as np

from .base import BaseRecipe


@dataclass
class IXIPrepareConfig:
    modalities: Sequence[str] = ("T1", "T2")
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    slice_axis: int = 2
    slice_range: Tuple[int, int] | None = None
    normalize: str = "minmax_per_slice"
    export_format: str = "npy"
    seed: int = 42


class IXIRecipe(BaseRecipe):
    """
    IXI dataset preparation recipe.

    Expected final raw layout:
        raw_root/
            T1/
                IXI002-Guys-0828-T1.nii.gz
                ...
            T2/
                IXI002-Guys-0828-T2.nii.gz
                ...
            PD/
                IXI002-Guys-0828-PD.nii.gz
                ...

    Supports:
        - optional automatic download from official IXI tar archives
        - subject-level train/val/test split
        - 2D slice export to .npy
        - manifest generation
    """

    name = "ixi"
    supports_download = True

    DOWNLOAD_URLS: Dict[str, str] = {
        "T1": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar",
        "T2": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar",
        "PD": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-PD.tar",
        "MRA": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-MRA.tar",
        "DTI": "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-DTI.tar",
    }

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
            "name": "IXI",
            "auto_download": self.supports_download,
            "modalities": ["T1", "T2", "PD", "MRA", "DTI"],
            "expected_structure": {
                "raw_root": str(self.raw_root),
                "example": {
                    "T1": "T1/IXI002-Guys-0828-T1.nii.gz",
                    "T2": "T2/IXI002-Guys-0828-T2.nii.gz",
                    "PD": "PD/IXI002-Guys-0828-PD.nii.gz",
                },
            },
            "notes": (
                "IXI can be downloaded from official tar archives and organized "
                "under modality folders. Data usage should follow the official IXI terms."
            ),
        }

    def download(
        self,
        modalities: Sequence[str] = ("T1", "T2", "PD"),
        force: bool = False,
        cleanup_archives: bool = True,
    ) -> None:
        """
        Download IXI tar archives and organize files under raw_root/<MODALITY>/.

        Args:
            modalities: modalities to download, e.g. ("T1", "T2", "PD")
            force: redownload and overwrite organization if True
            cleanup_archives: remove temporary tar files after extraction
        """
        self.raw_root.mkdir(parents=True, exist_ok=True)

        requested = [m.upper() for m in modalities]
        unsupported = [m for m in requested if m not in self.DOWNLOAD_URLS]
        if unsupported:
            raise ValueError(
                f"Unsupported IXI download modalities: {unsupported}. "
                f"Supported: {sorted(self.DOWNLOAD_URLS.keys())}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for modality in requested:
                modality_dir = self.raw_root / modality
                modality_dir.mkdir(parents=True, exist_ok=True)

                if not force and self._modality_dir_has_nifti(modality_dir):
                    print(f"{modality} already exists in {modality_dir}, skipping download.")
                    continue

                archive_path = tmpdir / f"IXI-{modality}.tar"
                self._download_file(self.DOWNLOAD_URLS[modality], archive_path)
                self._extract_tar(archive_path, tmpdir / f"extract_{modality}")
                self._organize_extracted_nifti(
                    extracted_root=tmpdir / f"extract_{modality}",
                    modality=modality,
                    destination_dir=modality_dir,
                    force=force,
                )

                if cleanup_archives and archive_path.exists():
                    archive_path.unlink()

    def verify(self) -> None:
        if not self.raw_root.exists():
            raise FileNotFoundError(f"raw_root does not exist: {self.raw_root}")

        if not self.raw_root.is_dir():
            raise ValueError(f"raw_root is not a directory: {self.raw_root}")

        modality_dirs = [p for p in self.raw_root.iterdir() if p.is_dir()]
        if not modality_dirs:
            raise ValueError(
                f"No modality directories found under raw_root: {self.raw_root}"
            )

        found_any_nifti = False
        for modality_dir in modality_dirs:
            if self._modality_dir_has_nifti(modality_dir):
                found_any_nifti = True
                break

        if not found_any_nifti:
            raise ValueError(
                f"No NIfTI files found under modality folders in: {self.raw_root}"
            )

    def discover(self) -> List[Dict[str, Any]]:
        """
        Discover subjects that exist in all available modality folders.

        Returns:
            [
                {
                    "subject_id": "IXI002-Guys-0828",
                    "paths": {
                        "T1": "...",
                        "T2": "..."
                    }
                }
            ]
        """
        subjects_by_modality: Dict[str, Dict[str, Path]] = {}

        modality_dirs = [p for p in self.raw_root.iterdir() if p.is_dir()]
        for modality_dir in sorted(modality_dirs):
            modality = modality_dir.name.upper()
            files = self._list_nifti_files(modality_dir)

            subject_map: Dict[str, Path] = {}
            for file_path in files:
                subject_id = self._extract_subject_id(file_path.name, modality)
                if subject_id is None:
                    continue
                subject_map[subject_id] = file_path

            if subject_map:
                subjects_by_modality[modality] = subject_map

        if not subjects_by_modality:
            raise ValueError("No modality files were discovered.")

        return self._build_subject_entries(subjects_by_modality)

    def assign_splits(
        self,
        items: List[Dict[str, Any]],
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if len(split_ratio) != 3:
            raise ValueError("split_ratio must have length 3: (train, val, test)")

        if not np.isclose(sum(split_ratio), 1.0):
            raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

        items = list(items)
        rng = random.Random(seed)
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        return {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        }

    def prepare(
        self,
        config: IXIPrepareConfig | None = None,
        download: bool = False,
        download_modalities: Sequence[str] | None = None,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        config = config or IXIPrepareConfig()

        if config.export_format.lower() != "npy":
            raise ValueError(
                f"Only export_format='npy' is supported in IXIRecipe v1, got {config.export_format}"
            )

        if download:
            requested_download_modalities = (
                tuple(download_modalities)
                if download_modalities is not None
                else tuple(config.modalities)
            )
            self.download(
                modalities=requested_download_modalities,
                force=force_download,
            )

        self.verify()
        self.ensure_roots()

        subjects = self.discover()
        filtered_subjects = self._filter_subjects_by_modalities(subjects, config.modalities)

        if not filtered_subjects:
            raise ValueError(
                f"No subjects found containing all requested modalities: {config.modalities}"
            )

        split_subjects = self.assign_splits(
            filtered_subjects,
            split_ratio=config.split_ratio,
            seed=config.seed,
        )

        split_entries: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for split_name, subject_items in split_subjects.items():
            for subject in subject_items:
                entries = self._export_subject(
                    subject=subject,
                    split=split_name,
                    config=config,
                )
                split_entries[split_name].extend(entries)

        manifest_paths = self.write_split_manifests(split_entries)
        self.save_config(config)

        return {
            "recipe": self.name,
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_subject_entries(
        self,
        subjects_by_modality: Dict[str, Dict[str, Path]],
    ) -> List[Dict[str, Any]]:
        modality_names = list(subjects_by_modality.keys())
        common_subjects = set(subjects_by_modality[modality_names[0]].keys())

        for modality in modality_names[1:]:
            common_subjects &= set(subjects_by_modality[modality].keys())

        entries: List[Dict[str, Any]] = []
        for subject_id in sorted(common_subjects):
            paths = {
                modality: str(subjects_by_modality[modality][subject_id])
                for modality in modality_names
            }
            entries.append(
                {
                    "subject_id": subject_id,
                    "paths": paths,
                }
            )
        return entries

    def _filter_subjects_by_modalities(
        self,
        subjects: List[Dict[str, Any]],
        modalities: Sequence[str],
    ) -> List[Dict[str, Any]]:
        required = {m.upper() for m in modalities}
        filtered = []

        for subject in subjects:
            available = set(subject["paths"].keys())
            if required.issubset(available):
                filtered.append(
                    {
                        "subject_id": subject["subject_id"],
                        "paths": {m.upper(): subject["paths"][m.upper()] for m in modalities},
                    }
                )

        return filtered

    def _export_subject(
        self,
        subject: Dict[str, Any],
        split: str,
        config: IXIPrepareConfig,
    ) -> List[Dict[str, Any]]:
        subject_id = subject["subject_id"]

        volumes: Dict[str, np.ndarray] = {}
        for modality, path_str in subject["paths"].items():
            vol = nib.load(path_str).get_fdata()
            volumes[modality] = vol

        ref_shape = next(iter(volumes.values())).shape
        for modality, vol in volumes.items():
            if vol.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch for subject {subject_id}: "
                    f"{modality} has shape {vol.shape}, expected {ref_shape}"
                )

        axis = config.slice_axis
        num_slices = ref_shape[axis]
        start, end = self._resolve_slice_range(config.slice_range, num_slices)

        entries: List[Dict[str, Any]] = []

        for slice_idx in tqdm(
            range(start, end),
            desc=f"{subject_id} slices",
            leave=False,
        ):
            saved_paths: Dict[str, str] = {}

            for modality, volume in volumes.items():
                slice_data = self._extract_slice(volume, axis=axis, index=slice_idx)
                slice_data = self._normalize_slice(slice_data, method=config.normalize)

                out_dir = self.ensure_dir(self.prepared_root / modality / split)
                out_path = out_dir / f"{subject_id}_slice_{slice_idx:03d}.npy"
                np.save(out_path, slice_data.astype(np.float32))

                saved_paths[modality] = self.to_relative_path(out_path, self.prepared_root)

            entry = {
                "id": f"{subject_id}_slice_{slice_idx:03d}",
                "subject_id": subject_id,
                "split": split,
                "paths": saved_paths,
                "meta": {
                    "dataset": "IXI",
                    "slice_index": slice_idx,
                    "slice_axis": axis,
                    "modalities": [m.upper() for m in config.modalities],
                },
            }
            entries.append(entry)

        return entries

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
            raise ValueError(
                f"Invalid slice_range after clamping: ({start}, {end}) for num_slices={num_slices}"
            )

        return start, end

    def _extract_slice(
        self,
        volume: np.ndarray,
        axis: int,
        index: int,
    ) -> np.ndarray:
        if axis == 0:
            return volume[index, :, :]
        if axis == 1:
            return volume[:, index, :]
        if axis == 2:
            return volume[:, :, index]
        raise ValueError(f"slice_axis must be 0, 1, or 2, got {axis}")

    def _normalize_slice(
        self,
        slice_data: np.ndarray,
        method: str = "minmax_per_slice",
    ) -> np.ndarray:
        if method == "none":
            return slice_data.astype(np.float32)

        if method == "minmax_per_slice":
            slice_data = slice_data.astype(np.float32)
            min_val = float(slice_data.min())
            max_val = float(slice_data.max())
            return (slice_data - min_val) / (max_val - min_val + 1e-8)

        raise ValueError(f"Unsupported normalize method: {method}")

    def _extract_subject_id(self, filename: str, modality: str) -> str | None:
        suffix_nii = f"-{modality}.nii"
        suffix_niigz = f"-{modality}.nii.gz"

        if filename.endswith(suffix_niigz):
            return filename[: -len(suffix_niigz)]
        if filename.endswith(suffix_nii):
            return filename[: -len(suffix_nii)]
        return None

    def _list_nifti_files(self, directory: Path) -> List[Path]:
        return sorted(list(directory.glob("*.nii")) + list(directory.glob("*.nii.gz")))

    def _modality_dir_has_nifti(self, modality_dir: Path) -> bool:
        return len(self._list_nifti_files(modality_dir)) > 0
    
    
    # def _download_file(self, url: str, destination: Path) -> None:
    #     destination.parent.mkdir(parents=True, exist_ok=True)
    #     urllib.request.urlretrieve(url, destination)
    

    def _download_file(self, url: str, destination: Path) -> None:
        """
        Download file with progress bar.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)

        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))

            with open(destination, "wb") as f, tqdm(
                desc=f"Downloading {destination.name}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:

                while True:
                    chunk = response.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break

                    f.write(chunk)
                    bar.update(len(chunk))
    

    # def _extract_tar(self, archive_path: Path, extract_dir: Path) -> None:
    #     extract_dir.mkdir(parents=True, exist_ok=True)
    #     with tarfile.open(archive_path, "r") as tar:
    #         tar.extractall(path=extract_dir)

    def _extract_tar(self, archive_path: Path, extract_dir: Path) -> None:
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r") as tar:
            members = tar.getmembers()

            for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                tar.extract(member, path=extract_dir)

    def _organize_extracted_nifti(
        self,
        extracted_root: Path,
        modality: str,
        destination_dir: Path,
        force: bool = False,
    ) -> None:
        """
        Move/copy extracted IXI modality files into raw_root/<MODALITY>/.
        """
        nifti_files = sorted(list(extracted_root.rglob("*.nii")) + list(extracted_root.rglob("*.nii.gz")))
        if not nifti_files:
            raise ValueError(
                f"No NIfTI files found after extracting archive for modality {modality}"
            )

        for src in nifti_files:
            dst = destination_dir / src.name
            if dst.exists():
                if force:
                    dst.unlink()
                else:
                    continue
            shutil.move(str(src), str(dst))