from abc import ABC, abstractmethod
from typing import Any

import lightning as L
from loguru import logger
from omegaconf import DictConfig
from torch import optim
from lightning.pytorch.utilities.types import STEP_OUTPUT


class BaseLightningModel(L.LightningModule, ABC):
    def __init__(self, predictor_config: DictConfig, lr: float, gene_names: list[str]):
        super().__init__()
        self.lr = lr
        self.gene_names = gene_names
        self._prepare_metrics()
        self.test_mode = "bag"

        self.save_hyperparameters()

    @abstractmethod
    def _prepare_metrics(self) -> None:
        pass

    @abstractmethod
    def loop_step(self, batch: Any, split: str) -> STEP_OUTPUT:
        pass

    @abstractmethod
    def on_epoch_end(self, split: str) -> None:
        pass

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return self.loop_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return self.loop_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self.on_epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self.on_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        logger.info("Computing final metrics...")
        self.on_epoch_end("test")

    @abstractmethod
    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}

    def set_test_mode(self, mode: str) -> None:
        self.test_mode = mode
        logger.info(f"Set test mode to {mode}")
