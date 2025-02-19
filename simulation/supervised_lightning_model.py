from typing import Any

import lightning as L
from loguru import logger
from omegaconf import DictConfig
from torch import optim
from lightning.pytorch.utilities.types import STEP_OUTPUT

from scellst.constant import REGISTRY_KEYS
from scellst.type import TaskType
from simulation.supervised_model import SupervisedModel, DistributionSupervisedModel


class SupervisedLightningModel(L.LightningModule):
    def __init__(
        self,
        task_type: str,
        predictor_config: DictConfig,
        lr: float,
        gene_names: list[str],
        criterion: str,
    ):
        super().__init__()
        # Create model
        if task_type == TaskType.regression:
            self.model = SupervisedModel(
                predictor_config=predictor_config,
                criterion=criterion,
            )
        elif task_type in [TaskType.nb_mean_regression, TaskType.nb_total_regression]:
            self.model = DistributionSupervisedModel(
                predictor_config=predictor_config,
                criterion="nll",
                task_type=task_type,
            )
        else:
            raise ValueError(
                f"Unknown predictor type {task_type}, should be in {TaskType.list()}"
            )
        self.lr = lr
        self.gene_names = gene_names

        self.save_hyperparameters()
        logger.info(self)

    def loop_step(self, batch: Any, split: str) -> STEP_OUTPUT:
        output_dict = self.model(batch)
        loss = self.model.loss(output_dict, batch)
        self.log(
            f"{split}_loss",
            loss,
            batch_size=len(batch[REGISTRY_KEYS.Y_INS_KEY]),
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

    def predict_step(self, batch: Any) -> STEP_OUTPUT:
        # Check whether the model is used for spots or cells
        return self.model(batch)[REGISTRY_KEYS.OUTPUT_PREDICTION]

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
