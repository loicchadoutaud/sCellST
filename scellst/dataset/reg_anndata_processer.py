import logging

from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

from scellst.dataset.base_anndata_processer import BaseAnndataProcessor
from scellst.type import TaskType


logger = logging.getLogger(__name__)


class RegDatasetProcesser(BaseAnndataProcessor):
    """
    Data preparation class for regression tasks.
    """
    def __init__(
        self,
        task_type: str,
        n_folds: int,
        batch_size: int,
        dataset_type: str,
        folder_info: str | None = None,
        model_type: str = "mil",
        scale_labels: bool = False,
    ) -> None:
        super().__init__(
            task_type,
            n_folds,
            batch_size,
            dataset_type,
            folder_info,
            model_type
        )
        self.scale_labels = scale_labels

    def _set_labels(self, adata: AnnData) -> AnnData:
        """Method to prepare output labels."""
        if self.task_type in [
            TaskType.nb_total_regression,
            TaskType.nb_mean_regression,
        ]:
            adata.layers["target"] = adata.layers["counts"]
        elif self.task_type == TaskType.regression:
            adata.layers["target"] = adata.X
        else:
            raise ValueError(f"{self.task_type} must be in {TaskType.list()}")
        # Convert to numpy array
        if isinstance(adata.layers["target"], csr_matrix):
            adata.layers["target"] = adata.layers["target"].toarray()
        return adata

    def _scale_labels(self, adata: AnnData) -> AnnData:
        """Method to scale output labels."""
        logger.info("Scaling expression values.")
        self.scaler = MinMaxScaler()
        self.scaler.fit(adata.layers["target"][self.split_dict["train"]])
        adata.layers["target"] = self.scaler.transform(adata.layers["target"])
        return adata

    def _transform_labels(self, adata: AnnData) -> None:
        """Method to apply all transformations to output labels."""
        # Scale values
        if self.scale_labels:
            self._scale_labels(adata)
