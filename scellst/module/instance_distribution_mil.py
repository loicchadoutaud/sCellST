from abc import ABC
from typing import Callable

import torch
from torch import Tensor
from .base_mil_model import BaseMilModel
from .scatter_code import scatter
from .. import REGISTRY_KEYS
from scellst.module.distributions import NegativeBinomial
from scellst.module.predictor import DistributionPredictor
from ..type import TaskType


class InstanceDistributionMilModel(BaseMilModel, ABC):
    def __init__(
        self,
        gene_predictor: DistributionPredictor,
        criterion: Callable,
        aggregation_type: str,
        parametrisation: str,
    ) -> None:
        super().__init__(gene_predictor, criterion, aggregation_type)
        self.parametrisation = parametrisation

    def create_nb_distribution(
        self, dist_param: Tensor, gene_param: Tensor
    ) -> NegativeBinomial:
        if self.parametrisation == TaskType.nb_mean_regression:
            return NegativeBinomial(mu=dist_param, theta=torch.exp(gene_param))
        elif self.parametrisation == TaskType.nb_total_regression:
            return NegativeBinomial(total_count=dist_param, logits=gene_param)
        else:
            raise ValueError("This should not happen !")

    def predict_bag(
        self,
        batch_data: dict[str, Tensor | NegativeBinomial],
        instance_dict: dict[str, Tensor | NegativeBinomial],
    ) -> dict[str, Tensor | NegativeBinomial]:
        # Get parameter to aggregate
        bag_predictions = scatter(
            src=instance_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
            index=batch_data[REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY],
            dim=0,
            reduce=self.reduce,
        )
        bag_predictions = self.transformation(bag_predictions)
        bag_predictions = (
            bag_predictions.unsqueeze(1)
            if len(bag_predictions.shape) == 1
            else bag_predictions
        )
        bag_predictions = batch_data[REGISTRY_KEYS.LIBRARY_KEY] * bag_predictions

        # Get gene param
        unique, idx, counts = torch.unique(
            batch_data[REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
        idxs = idx[cum_sum]
        gene_param = instance_dict[REGISTRY_KEYS.OUTPUT_GENE_PARAM][idxs]

        # Prepare output dict
        bag_distribution = self.create_nb_distribution(bag_predictions, gene_param)
        return {
            REGISTRY_KEYS.OUTPUT_PREDICTION: bag_distribution.mean,
            REGISTRY_KEYS.OUTPUT_DISTRIBUTION: bag_distribution,
        }

    def loss(
        self,
        bag_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor | NegativeBinomial],
    ) -> Tensor:
        return self.criterion(
            bag_dict[REGISTRY_KEYS.OUTPUT_DISTRIBUTION],
            batch_data[REGISTRY_KEYS.Y_BAG_KEY],
        )


