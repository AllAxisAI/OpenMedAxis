from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Minimal reusable base dataset for OpenMedAxis.

    This class does not enforce any particular storage format.
    It only provides common utilities such as:
        - optional padding
        - optional normalization
        - optional dict-style outputs

    Subclasses are responsible for:
        - discovering files / records
        - implementing __len__
        - implementing __getitem__
    """

    def __init__(
        self,
        image_size: int | None = None,
        norm: bool = True,
        padding: bool = True,
        return_dict: bool = True,
    ) -> None:
        self.image_size = image_size
        self.norm = norm
        self.padding = padding
        self.return_dict = return_dict

        self.original_shape: tuple[int, int] | None = None

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def _pad_data(self, data: np.ndarray) -> np.ndarray:
        """
        Pad array on its last two dimensions to image_size x image_size.

        Supports arrays such as:
            (H, W)
            (N, H, W)
            (C, H, W)
            (..., H, W)
        """
        if self.image_size is None:
            return data

        h, w = data.shape[-2:]
        if h > self.image_size or w > self.image_size:
            raise ValueError(
                f"Cannot pad data with spatial shape {(h, w)} "
                f"to smaller image_size={self.image_size}"
            )

        pad_top = (self.image_size - h) // 2
        pad_bottom = self.image_size - h - pad_top
        pad_left = (self.image_size - w) // 2
        pad_right = self.image_size - w - pad_left

        pad_spec = [(0, 0)] * data.ndim
        pad_spec[-2] = (pad_top, pad_bottom)
        pad_spec[-1] = (pad_left, pad_right)

        return np.pad(data, pad_spec, mode="constant")

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Legacy normalization used in previous projects:
            [0, 1] -> [-1, 1]
        """
        return (data - 0.5) / 0.5

    def _prepare_2d(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare a single 2D image:
            - cast to float32
            - optional pad
            - optional normalize
            - add channel dimension => (1, H, W)
        """
        data = data.astype(np.float32)

        if self.padding:
            data = self._pad_data(data)

        if self.norm:
            data = self._normalize(data)

        data = np.expand_dims(data, axis=0)
        return data

    def _prepare_sample_dict(
        self,
        *,
        index: int,
        sample_id: str | int,
        source: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        sample: Dict[str, Any] = {
            "index": index,
            "id": sample_id,
            "meta": meta or {},
        }

        if source is not None:
            sample["source"] = source
        if target is not None:
            sample["target"] = target
        if image is not None:
            sample["image"] = image

        sample.update(extra)
        return sample