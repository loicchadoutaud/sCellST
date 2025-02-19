import torch
from omegaconf import DictConfig
from torch import nn, Tensor

from scellst.constant import REGISTRY_KEYS
from scellst.model.base_mil_model import loss_dict
from scellst.module.distributions import NegativeBinomial
from scellst.module.gene_predictor import GenePredictor, GeneDistributionPredictor
from scellst.type import TaskType


class SupervisedModel(nn.Module):
    def __init__(
        self,
        predictor_config: DictConfig,
        criterion: str,
    ):
        """
        Base module.

        Args:
            gene_predictor: neural network to predict a score for each instance / bag
            criterion: loss function to compare predicted score to true label
            aggregation_type: aggregation function to compute a score for each bag based on the score of each instance
        """
        super().__init__()
        self.gene_predictor = GenePredictor(**predictor_config)
        self.output_dim = self.gene_predictor.output_dim
        assert (
            criterion in loss_dict.keys()
        ), f"criterion must be in {loss_dict.keys()}, got {criterion}."
        self.criterion = loss_dict[criterion]

    def loss(
        self,
        output_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor],
    ) -> Tensor:
        return self.criterion(
            output_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
            batch_data[REGISTRY_KEYS.Y_INS_KEY],
        )

    def forward(
        self,
        batch_data: dict[str, Tensor],
    ) -> Tensor:
        batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING] = batch_data[REGISTRY_KEYS.X_KEY]
        return self.gene_predictor(batch_data)


class DistributionSupervisedModel(nn.Module):
    def __init__(self, predictor_config: DictConfig, criterion: str, task_type: str):
        """
        Base module.

        Args:
            gene_predictor: neural network to predict a score for each instance / bag
            criterion: loss function to compare predicted score to true label
            aggregation_type: aggregation function to compute a score for each bag based on the score of each instance
        """
        super().__init__()
        self.gene_predictor = GeneDistributionPredictor(**predictor_config)
        self.output_dim = self.gene_predictor.output_dim
        assert (
            criterion in loss_dict.keys()
        ), f"criterion must be in {loss_dict.keys()}, got {criterion}."
        self.criterion = loss_dict[criterion]
        self.task_type = task_type

    def create_nb_distribution(
        self, dist_param: Tensor, gene_param: Tensor
    ) -> NegativeBinomial:
        if self.task_type == TaskType.nb_mean_regression:
            return NegativeBinomial(mu=dist_param, theta=torch.exp(gene_param))
        elif self.task_type == TaskType.nb_total_regression:
            return NegativeBinomial(total_count=dist_param, logits=gene_param)
        else:
            raise ValueError("This should not happen !")

    def loss(
        self,
        output_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor],
    ) -> Tensor:
        distribution = self.create_nb_distribution(
            dist_param=output_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
            gene_param=output_dict[REGISTRY_KEYS.OUTPUT_GENE_PARAM],
        )
        return self.criterion(
            distribution,
            batch_data[REGISTRY_KEYS.Y_INS_KEY],
        )

    def forward(
        self,
        batch_data: dict[str, Tensor],
    ) -> Tensor:
        batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING] = batch_data[REGISTRY_KEYS.X_KEY]
        return self.gene_predictor(batch_data)
