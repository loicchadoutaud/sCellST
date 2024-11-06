import logging
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn, Tensor

from scellst import REGISTRY_KEYS
from scellst.module.distributions import NegativeBinomial
from scellst.module.predictor import BasePredictor
from scellst.type import AggType

logger = logging.getLogger(__name__)


class BaseMilModel(nn.Module, ABC):
    def __init__(
        self,
        gene_predictor: BasePredictor,
        criterion: Callable,
        aggregation_type: str,
    ):
        """
        Base module.

        Args:
            gene_predictor: neural network to predict a score for each instance / bag
            criterion: loss function to compare predicted score to true label
            aggregation_type: aggregation function to compute a score for each bag based on the score of each instance
        """
        super().__init__()
        self.gene_predictor = gene_predictor
        self.output_dim = self.gene_predictor.output_dim
        self.criterion = criterion
        self.agg_type = aggregation_type

        self.reduce, self.transformation = self._define_aggregation_function()
        logger.info(self)

    def _define_aggregation_function(self) -> tuple[str, Callable | nn.Module]:
        """
        Define aggregation function to compute a score for each bag based on the score of each instance.
        Returns:
            aggregation function
        """
        if self.agg_type == AggType.mean:
            return "mean", nn.Identity()
        elif self.agg_type == AggType.sum:
            return "sum", nn.Identity()
        elif self.agg_type == AggType.max:
            return "max", nn.Identity()
        elif self.agg_type == AggType.log_mean:
            return "mean", torch.log1p
        elif self.agg_type == AggType.log_sum:
            return "sum", torch.log1p
        else:
            raise ValueError(
                f"Unknown aggregation type {self.agg_type},\nMust be one of {AggType.list()}"
            )

    def get_latent_dim(self) -> int:
        return self.gene_predictor.get_latent_dim()

    @abstractmethod
    def predict_bag(
        self,
        batch_data: dict[str, Tensor | NegativeBinomial],
        instance_dict: dict[str, Tensor | NegativeBinomial],
    ) -> dict[str, Tensor | NegativeBinomial]:
        pass

    @abstractmethod
    def loss(
        self,
        bag_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor | NegativeBinomial],
    ) -> Tensor:
        pass

    def forward(
        self,
        batch_data: dict[str, Tensor],
    ) -> tuple[
        dict[str, Tensor | NegativeBinomial], dict[str, Tensor | NegativeBinomial]
    ]:
        # Compute embedding for each instances of all bags
        batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING] = batch_data[REGISTRY_KEYS.X_KEY]

        # Compute class/score prediction for each instance
        instance_dict = self.gene_predictor(batch_data)

        # Compute predictions for each bag
        bag_dict = self.predict_bag(batch_data, instance_dict)

        return bag_dict, instance_dict

    def predict_instance(self, batch_data: dict[str, Tensor]) -> Tensor:
        """Predict function for instance data."""
        batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING] = batch_data[REGISTRY_KEYS.X_KEY]
        return self.gene_predictor(batch_data)
