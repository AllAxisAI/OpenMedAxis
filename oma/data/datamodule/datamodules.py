from __future__ import annotations

from typing import Any, Dict, Optional, Type

import lightning as L
from torch.utils.data import DataLoader, Dataset


class LSplitDataModule(L.LightningDataModule):
    """
    Lightning-compatible split-based DataModule.

    This module assumes the dataset class can be instantiated separately
    for different splits, typically with an argument such as:
        stage="train" / "val" / "test"

    Example:
        dm = LSplitDataModule(
            dataset_cls=NumpyDataset,
            dataset_kwargs={
                "data_dir": "...",
                "source_modality": "T1",
                "target_modality": "T2",
                "image_size": 256,
                "padding": True,
                "norm": True,
            },
            train_dataloader_kwargs={"batch_size": 8, "shuffle": True, "drop_last": True},
            val_dataloader_kwargs={"batch_size": 8, "shuffle": False},
            test_dataloader_kwargs={"batch_size": 8, "shuffle": False},
        )
    """

    def __init__(
        self,
        dataset_cls: Type[Dataset],
        dataset_kwargs: Dict[str, Any],
        split_arg: str = "stage",
        train_split: str = "train",
        val_split: str = "val",
        test_split: str = "test",
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        val_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dict(dataset_kwargs)

        self.split_arg = split_arg
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.train_dataloader_kwargs = {
            "batch_size": 1,
            "shuffle": True,
            "drop_last": True,
            "num_workers": 0,
        }
        if train_dataloader_kwargs is not None:
            self.train_dataloader_kwargs.update(train_dataloader_kwargs)

        self.val_dataloader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 0,
        }
        if val_dataloader_kwargs is not None:
            self.val_dataloader_kwargs.update(val_dataloader_kwargs)

        self.test_dataloader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 0,
        }
        if test_dataloader_kwargs is not None:
            self.test_dataloader_kwargs.update(test_dataloader_kwargs)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.save_hyperparameters(ignore=["dataset_cls"])

    def _make_dataset(self, split_name: str) -> Dataset:
        kwargs = dict(self.dataset_kwargs)
        kwargs[self.split_arg] = split_name
        return self.dataset_cls(**kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Lightning stages:
            - None
            - 'fit'
            - 'validate'
            - 'test'
            - 'predict' (not implemented here)
        """
        if stage is None:
            if self.train_dataset is None:
                self.train_dataset = self._make_dataset(self.train_split)
            if self.val_dataset is None:
                self.val_dataset = self._make_dataset(self.val_split)
            if self.test_dataset is None:
                self.test_dataset = self._make_dataset(self.test_split)
            return

        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self._make_dataset(self.train_split)
            if self.val_dataset is None:
                self.val_dataset = self._make_dataset(self.val_split)

        elif stage == "validate":
            if self.val_dataset is None:
                self.val_dataset = self._make_dataset(self.val_split)

        elif stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self._make_dataset(self.test_split)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is not initialized. Call setup('fit') first."
            )
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError(
                "val_dataset is not initialized. Call setup('fit') or setup('validate') first."
            )
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "test_dataset is not initialized. Call setup('test') first."
            )
        return DataLoader(self.test_dataset, **self.test_dataloader_kwargs)