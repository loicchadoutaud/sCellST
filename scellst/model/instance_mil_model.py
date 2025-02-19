from omegaconf import DictConfig
from torch import Tensor
from torch_scatter import scatter

from scellst.constant import REGISTRY_KEYS
from scellst.model.base_mil_model import BaseMilModel
from scellst.module.distributions import NegativeBinomial
from scellst.module.gene_predictor import GenePredictor


class InstanceMilModel(BaseMilModel):
    def __init__(
        self,
        predictor_config: DictConfig,
        criterion: str,
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
            criterion,
            aggregation_type,
        )
        self.gene_predictor = GenePredictor(**predictor_config)
        self.output_dim = self.gene_predictor.output_dim

    def predict_instance(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor | NegativeBinomial]:
        """Predict function for instance data."""
        batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING] = batch_data[REGISTRY_KEYS.X_KEY]
        return self.gene_predictor(batch_data)

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
