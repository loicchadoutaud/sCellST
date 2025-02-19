from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torch.distributions import Distribution

from scellst.module.distributions import NegativeBinomial


class BaseMilModel(nn.Module, ABC):
    def __init__(
        self,
        criterion: str,
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
        assert (
            criterion in loss_dict.keys()
        ), f"criterion must be in {loss_dict.keys()}, got {criterion}."
        self.criterion = loss_dict[criterion]
        self.reduce = aggregation_type

    @abstractmethod
    def loss(
        self,
        bag_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor | NegativeBinomial],
    ) -> Tensor:
        pass

    @abstractmethod
    def predict_instance(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor | NegativeBinomial]:
        pass

    @abstractmethod
    def predict_bag(
        self,
        batch_data: dict[str, Tensor | NegativeBinomial],
        instance_dict: dict[str, Tensor | NegativeBinomial],
    ) -> dict[str, Tensor | NegativeBinomial]:
        pass

    def forward(
        self,
        batch_data: dict[str, Tensor],
    ) -> tuple[
        dict[str, Tensor | NegativeBinomial], dict[str, Tensor | NegativeBinomial]
    ]:
        # Compute class/score prediction for each instance
        instance_dict = self.predict_instance(batch_data)

        # Compute predictions for each bag
        bag_dict = self.predict_bag(batch_data, instance_dict)

        return bag_dict, instance_dict


def negative_log_likelihood(output: Distribution, target: Tensor) -> Tensor:
    return -output.log_prob(target.int()).sum(-1).mean()


loss_dict = {"mse": torch.nn.functional.mse_loss, "nll": negative_log_likelihood}
