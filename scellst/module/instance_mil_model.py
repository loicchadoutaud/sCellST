import logging
from typing import Callable

from torch import Tensor

from .distributions import NegativeBinomial
from .base_mil_model import BaseMilModel
from ..constants import REGISTRY_KEYS
from .predictor import BasePredictor
from .scatter_code import scatter

logger = logging.getLogger(__name__)


class InstanceMilModel(BaseMilModel):
    def __init__(
        self,
        gene_predictor: BasePredictor,
        criterion: Callable,
        aggregation_type: str,
    ):
        """
        Multiple Instance Learning module.

        Args:
            gene_predictor: neural network to predict a score for each instance
            criterion: loss function to compare predicted score to true label
           , aggregation_type: aggregation function to compute a score for each bag based on the score of each instance
        """
        super().__init__(
            gene_predictor,
            criterion,
            aggregation_type,
        )

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

        # Prepare output dict
        return {
            REGISTRY_KEYS.OUTPUT_PREDICTION: bag_predictions,
        }

    def loss(
        self,
        bag_dict: dict[str, Tensor | NegativeBinomial],
        batch_data: dict[str, Tensor | NegativeBinomial],
    ) -> Tensor:
        return self.criterion(
            bag_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
            batch_data[REGISTRY_KEYS.Y_BAG_KEY],
        )
