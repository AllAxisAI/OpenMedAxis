from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseDataModule(ABC):
    """
    Framework-native datamodule abstraction for OpenMedAxis.

    This class does not depend on Lightning.
    It provides a familiar lifecycle for dataset setup and dataloader creation.
    """

    def __init__(self) -> None:
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """
        Optional hook for downloading, verifying, or preparing data.
        """
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build datasets for the requested stage.

        stage can be:
            - None
            - 'fit'
            - 'validate'
            - 'test'
            - 'predict'
        """
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement train_dataloader()"
        )

    def val_dataloader(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement val_dataloader()"
        )

    def test_dataloader(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement test_dataloader()"
        )

    def predict_dataloader(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_dataloader()"
        )