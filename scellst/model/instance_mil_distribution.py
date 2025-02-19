import torch
from omegaconf import DictConfig
from torch import Tensor
from torch_scatter import scatter

from scellst.constant import REGISTRY_KEYS
from scellst.model.base_mil_model import BaseMilModel
from scellst.module.distributions import NegativeBinomial
from scellst.module.gene_predictor import GeneDistributionPredictor
from scellst.type import TaskType


class InstanceDistributionMilModel(BaseMilModel):
    def __init__(
        self,
        predictor_config: DictConfig,
        criterion: str,
        aggregation_type: str,
        task_type: str,
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
        self.gene_predictor = GeneDistributionPredictor(**predictor_config)
        self.output_dim = self.gene_predictor.output_dim
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
        raw_bag_predictions = scatter(
            src=instance_dict[REGISTRY_KEYS.OUTPUT_PREDICTION],
            index=batch_data[REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY],
            dim=0,
            reduce=self.reduce,
        )
        raw_bag_predictions = (
            raw_bag_predictions.unsqueeze(1)
            if len(raw_bag_predictions.shape) == 1
            else raw_bag_predictions
        )
        bag_predictions = batch_data[REGISTRY_KEYS.SIZE_FACTOR] * raw_bag_predictions

        # Get gene parameter
        unique, idx, counts = torch.unique(
            batch_data[REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
        gene_param = instance_dict[REGISTRY_KEYS.OUTPUT_GENE_PARAM][idx[cum_sum]]

        # Prepare output dict
        bag_distribution = self.create_nb_distribution(bag_predictions, gene_param)
        return {
            REGISTRY_KEYS.OUTPUT_PREDICTION: raw_bag_predictions,
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
