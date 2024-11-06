import logging
import os
from abc import ABC, abstractmethod

from anndata import AnnData
from numpy import ndarray
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader

from .mil_dataset import (
    EmbeddedMilDataset,
    EmbeddedInstanceDataset,
    CelltypeMilDataset,
    CelltypeInstanceDataset,
)
from .data_utils import (
    split_train_val_all,
)
from .dataset_utils import (
    create_dataloader_func_dict,
    create_dataloader_mil_instance,
    create_dataloader_mil_reg,
)
from .supervised_dataset import SupervisedDataset
from ..type import InputType, ModelType

logger = logging.getLogger(__name__)


class BaseAnndataProcessor(ABC):
    """
    Base abstract class for data preparation.
    This class does everything related to data preparation (input and output), take care of data splits
    and have helping functions for dataset and dataloader creation.
    """

    def __init__(
        self,
        task_type: str,
        n_folds: int,
        batch_size: int,
        dataset_type: str,
        folder_info: str | None = None,
        model_type: str = "mil"
    ):
        self.task_type = task_type
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.folder_info = folder_info
        self.model_type = model_type

        # Prepare dataset class
        match model_type:
            # Mil dataset case
            case ModelType.instance_mil:
                match dataset_type:
                    case InputType.cell_type:
                        self.spot_dataset_class = CelltypeMilDataset
                        self.instance_dataset_class = CelltypeInstanceDataset
                    case InputType.embedding:
                        self.spot_dataset_class = EmbeddedMilDataset
                        self.instance_dataset_class = EmbeddedInstanceDataset
                    case _:
                        raise ValueError(f"Wrong input type, got: {dataset_type}")
            # Supervised dataset case
            case ModelType.supervised:
                self.dataset_class = SupervisedDataset
            case _:
                raise ValueError(f"Wrong model type, got: {model_type}")

        self.Y_bags = None
        self.Y_instances = None
        self.scaler = None

        # Create dictionary to store training info
        self.split_dict = {}
        self.split_dict_names = {}
        self.normalisation_dict = None

    @abstractmethod
    def _set_labels(self, adata: AnnData) -> None:
        pass

    @abstractmethod
    def _transform_labels(self, adata: AnnData) -> None:
        pass

    def get_label_dim(self) -> int:
        return self.Y_bags.shape[1]

    def _create_mil_datasets(
        self,
        adata: AnnData,
        data_folder: str,
    ) -> tuple[Dataset, Dataset]:
        """Create spot datasets for training and inference."""
        train_dataset = self.spot_dataset_class(
            # Data info
            adata=adata[self.split_dict_names["train"]],
            instance_folder=os.path.join(data_folder, "mil", self.folder_info  + ".h5"),
        )
        val_dataset = self.spot_dataset_class(
            # Data info
            adata=adata[self.split_dict_names["val"]],
            instance_folder=os.path.join(data_folder, "mil", self.folder_info  + ".h5"),
        )

        return train_dataset, val_dataset

    def _create_supervised_datasets(
        self,
        adata: AnnData,
    ) -> tuple[Dataset, Dataset]:
        """Create spot datasets for training and inference."""
        train_dataset = self.dataset_class(
            # Data info
            adata=adata[self.split_dict_names["train"]],
        )
        val_dataset = self.dataset_class(
            # Data info
            adata=adata[self.split_dict_names["val"]],
        )

        return train_dataset, val_dataset

    def _create_mil_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> tuple[DataLoader, DataLoader]:
        """Create all dataloaders."""
        create_dataloader_func = create_dataloader_func_dict[self.task_type]
        train_dataloader = create_dataloader_func(
            dataset=train_dataset,
            batch_size=self.batch_size,
            train_mode=True,
        )
        val_dataloader = create_dataloader_func(
            dataset=val_dataset,
            batch_size=self.batch_size,
            train_mode=False,
        )
        return train_dataloader, val_dataloader

    def _create_supervised_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        return train_dataloader, val_dataloader

    def prepare_folds(self, adata: AnnData) -> None:
        """Perform splits"""
        # Prepare labels
        self._set_labels(adata)

        # Split spots between train and validation
        logger.info("Using random split in slide.")
        self.split_dict = split_train_val_all(
            adata,
            self.n_folds,
        )
        self.split_dict_names = {
            key: adata.obs_names[value] for key, value in self.split_dict.items()
        }
        logger.info(f"Fold - train set size: {len(self.split_dict_names['train'])}")
        logger.info(f"Fold - validation set size: {len(self.split_dict_names['val'])}")

    def prepare_mil_training_dataloaders(
        self, adata: AnnData, train_folder_path: str
    ) -> tuple[DataLoader, DataLoader]:
        # Transform labels
        self._transform_labels(adata)

        # Create datasets
        list_datasets = self._create_mil_datasets(adata, train_folder_path)

        # Create dataloader
        return self._create_mil_dataloaders(*list_datasets)

    def prepare_supervised_training_dataloaders(
        self, adata: AnnData, train_folder_path: str
    ) -> tuple[DataLoader, DataLoader]:
        # Transform labels
        self._transform_labels(adata)

        # Create datasets
        list_datasets = self._create_supervised_datasets(adata, train_folder_path)

        # Create dataloader
        return self._create_supervised_dataloaders(*list_datasets)

    def prepare_spot_inference_dataloader(
        self, adata: AnnData, data_folder: str, normalisation_path_dict: str
    ) -> DataLoader:
        if isinstance(adata.X, csr_matrix):
            adata.layers["target"] = adata.X.toarray()
        else:
            adata.layers["target"] = adata.X
        spot_dataset = self.spot_dataset_class(
            adata=adata,
            instance_folder=os.path.join(data_folder, "mil", self.folder_info  + ".h5"),
        )
        return create_dataloader_mil_reg(
            dataset=spot_dataset,
            batch_size=self.batch_size,
            train_mode=False,
        )

    def prepare_cell_inference_dataloader(
        self, cell_labels: ndarray, data_folder: str
    ) -> DataLoader:
        data_folder = os.path.join(data_folder, self.folder_info  + ".h5")
        cell_dataset = self.instance_dataset_class(
            # Data info
            instance_labels=cell_labels,
            instance_folder=data_folder,
        )
        return create_dataloader_mil_instance(cell_dataset)

    def prepare_cell_supervised_inference_dataloader(
        self, adata: AnnData
    ) -> DataLoader:
        if "target" not in adata.layers.keys():
            adata.layers["target"] = adata.X
        cell_dataset = self.dataset_class(
            adata=adata
        )
        return DataLoader(
            dataset=cell_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
