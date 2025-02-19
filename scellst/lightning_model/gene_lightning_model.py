from typing import Any

import torchmetrics
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from omegaconf import DictConfig
from scellst.constant import REGISTRY_KEYS
from scellst.model.instance_mil_model import InstanceMilModel
from scellst.model.instance_mil_distribution import InstanceDistributionMilModel
from scellst.lightning_model.base_lightning_model import BaseLightningModel
from scellst.type import TaskType


class GeneLightningModel(BaseLightningModel):
    def __init__(
        self,
        task_type: str,
        predictor_config: DictConfig,
        lr: float,
        gene_names: list[str],
        criterion: str,
    ):
        super().__init__(predictor_config, lr, gene_names)
        self.task_type = task_type

        # Create model
        if task_type == TaskType.regression:
            self.model = InstanceMilModel(
                predictor_config=predictor_config,
                criterion=criterion,
                aggregation_type="mean",
            )
        elif task_type == TaskType.bag_regression:
            self.model = BagMilModel(
                predictor_config=predictor_config,
                criterion=criterion,
                aggregation_type="mean",
            )
        elif task_type == TaskType.att_regression:
            self.model = AttentionMilModel(
                predictor_config=predictor_config,
                criterion=criterion,
                aggregation_type="mean",
            )
        elif task_type in [TaskType.nb_mean_regression, TaskType.nb_total_regression]:
            self.model = InstanceDistributionMilModel(
                predictor_config=predictor_config,
                criterion="nll",
                aggregation_type="mean",
                task_type=task_type,
            )
        else:
            raise ValueError(
                f"Unknown predictor type {task_type}, should be in {TaskType.list()}"
            )

        logger.info(self)

    def _prepare_metrics(self) -> None:
        metrics = torchmetrics.MetricCollection(
            {
                "mse": torchmetrics.regression.MeanSquaredError(
                    num_outputs=len(self.gene_names)
                ),
                "pcc": torchmetrics.regression.PearsonCorrCoef(
                    num_outputs=len(self.gene_names)
                ),
            },
        )
        self.train_metrics = metrics.clone(prefix="val/")
        self.valid_metrics = metrics.clone(prefix="val/")
        metrics.add_metrics(
            {
                "scc": torchmetrics.regression.SpearmanCorrCoef(
                    num_outputs=len(self.gene_names)
                )
            }
        )
        self.test_metrics = metrics.clone()
        self.metrics = {
            "train": self.train_metrics,
            "val": self.valid_metrics,
            "test": self.test_metrics,
        }

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
        if split in ["val"]:
            self.metrics[split].update(
                bag_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
                batch[REGISTRY_KEYS.Y_BAG_KEY],
            )
        return loss

    def on_epoch_end(self, split: str) -> None:
        # Compute and log metrics at epoch end
        if split == "val":
            epoch_metrics = self.metrics[split].compute()
            if len(self.gene_names) == 1:
                epoch_metrics = {
                    f"{key}/{next(iter(self.gene_names))}": value
                    for key, value in epoch_metrics.items()
                }
            else:
                epoch_metrics = {
                    f"{key}/{gene}": value[i]
                    for key, value in epoch_metrics.items()
                    for i, gene in enumerate(self.gene_names)
                }
            self.log_dict(epoch_metrics, on_step=False, on_epoch=True)
            self.metrics[split].reset()

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        match self.test_mode:
            case "bag":
                bag_dict, instance_dict = self.model(batch)
                self.test_metrics.update(
                    bag_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
                    batch[REGISTRY_KEYS.Y_BAG_KEY],
                )
            case "instance":
                instance_dict = self.model.predict_instance(batch)
                self.test_metrics.update(
                    instance_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
                    batch[REGISTRY_KEYS.Y_INS_KEY],
                )
            case _:
                raise ValueError(f"Unknown test mode {self.test_mode}")

    def predict_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        match self.test_mode:
            case "bag":
                bag_dict, _ = self.model(batch)
                return bag_dict[REGISTRY_KEYS.OUTPUT_PREDICTION]
            case "instance":
                instance_dict = self.model.predict_instance(batch)
                return instance_dict[REGISTRY_KEYS.OUTPUT_PREDICTION]
            case _:
                raise ValueError(f"Unknown test mode {self.test_mode}")
