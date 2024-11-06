import logging
from typing import Callable

from torch import nn, Tensor

from scellst import REGISTRY_KEYS
from scellst.module.distributions import NegativeBinomial
from scellst.module.predictor import BasePredictor

logger = logging.getLogger(__name__)


class SupervisedModel(nn.Module):
    def __init__(
        self,
        gene_predictor: BasePredictor,
        criterion: Callable,
    ) -> None:
        super().__init__()
        self.gene_predictor = gene_predictor
        self.output_dim = self.gene_predictor.output_dim
        self.criterion = criterion
        logger.info(self)

    def get_latent_dim(self) -> int:
        return self.gene_predictor.get_latent_dim()

    def loss(
        self,
        bag_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor],
    ) -> Tensor:
        return self.criterion(
            bag_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
            batch_data[REGISTRY_KEYS.Y_INS_KEY],
        )

    def forward(
        self,
        batch_data: dict[str, Tensor],
    ) -> Tensor:
        return self.gene_predictor(batch_data)
