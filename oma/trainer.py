from __future__ import annotations

from typing import Any, Optional

import lightning as L


class Trainer:
    """
    Thin OMA wrapper around PyTorch Lightning's Trainer.

    Philosophy
    ----------
    - Lightning handles execution and infrastructure.
    - OMA exposes a stable user-facing training entrypoint.
    - This wrapper stays intentionally small in v0.1.

    Example
    -------
    >>> trainer = Trainer(max_epochs=10, accelerator="cpu")
    >>> trainer.fit(method, datamodule=datamodule)
    """

    def __init__(self, **trainer_kwargs: Any) -> None:
        self._trainer = L.Trainer(**trainer_kwargs)

    def fit(self, method: L.LightningModule, datamodule: Optional[L.LightningDataModule] = None, **kwargs: Any) -> Any:
        return self._trainer.fit(method, datamodule=datamodule, **kwargs)

    def validate(
        self,
        method: L.LightningModule,
        datamodule: Optional[L.LightningDataModule] = None,
        **kwargs: Any,
    ) -> Any:
        return self._trainer.validate(method, datamodule=datamodule, **kwargs)

    def test(
        self,
        method: L.LightningModule,
        datamodule: Optional[L.LightningDataModule] = None,
        **kwargs: Any,
    ) -> Any:
        return self._trainer.test(method, datamodule=datamodule, **kwargs)

    def predict(
        self,
        method: L.LightningModule,
        datamodule: Optional[L.LightningDataModule] = None,
        **kwargs: Any,
    ) -> Any:
        return self._trainer.predict(method, datamodule=datamodule, **kwargs)

    @property
    def lightning_trainer(self) -> L.Trainer:
        """
        Access the underlying Lightning Trainer when advanced control is needed.
        """
        return self._trainer