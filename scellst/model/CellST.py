from typing import Any

import lightning as L
from torch import optim
from lightning.pytorch.utilities.types import STEP_OUTPUT

from scellst import REGISTRY_KEYS
from scellst.module.base_mil_model import BaseMilModel


class CellST(L.LightningModule):
    def __init__(self, model: BaseMilModel, lr: float, gene_names: list[str]):
        super().__init__()
        self.model = model
        self.lr = lr
        self.gene_names = gene_names

        self.save_hyperparameters()

    def loop_step(self, batch: Any, split: str) -> STEP_OUTPUT:
        bag_dict, instance_dict = self.model(batch)
        loss = self.model.loss(bag_dict, batch)
        self.log(
            f"{split}_loss",
            loss,
            batch_size=len(batch[REGISTRY_KEYS.Y_BAG_KEY]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return self.loop_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return self.loop_step(batch, "val")

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT:
        return self.loop_step(batch, "test")

    def predict_step(self, batch: Any) -> Any:
        # Check whether the model is used for spots or cells
        if REGISTRY_KEYS.Y_BAG_KEY in batch.keys():
            return self.model(batch)
        else:
            return self.model.predict_instance(batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1.e-5),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
