import logging
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from anndata import AnnData
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset

from ..constants import REGISTRY_KEYS

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    h5_key: str
    def __init__(
        self,
        adata: AnnData,
        instance_folder: str,
    ) -> None:
        super().__init__()
        self.spot_names = adata.obs_names.tolist()
        self.mapping_dict = adata.uns["MIL"]["spot_cell_map"]
        self.instance_folder = instance_folder

        # Prepare spot labels
        self.bag_labels = torch.from_numpy(adata.layers["target"])
        if len(self.bag_labels.shape) == 1:
            self.bag_labels = torch.unsqueeze(self.bag_labels, dim=1).float()
        self.library_size = (
            torch.from_numpy(adata.obs["total_counts"].values).unsqueeze(dim=1).float()
        )
        self.batch_idx = torch.from_numpy(adata.obs["slide_idx"].values).int()
        self.spot_idx = torch.from_numpy(adata.obs["spot_idx"].values).int()

        # Prepare instance labels
        unique_values, inverse = np.unique(
            adata.uns["MIL"]["cell_label"], return_inverse=True
        )
        self.n_class_instance = len(unique_values)
        self.instance_labels = torch.from_numpy(inverse)
        self.instance_class = adata.uns["MIL"]["cell_label"]
        if len(self.instance_labels.shape) == 1:
            self.instance_labels = torch.unsqueeze(self.instance_labels, dim=1)

    def __len__(self) -> int:
        return len(self.spot_names)

    def _open_hdf5(self):
        self._h5file = h5py.File(self.instance_folder, 'r')
        self._dataset = self._h5file[self.h5_key]

    @abstractmethod
    def get_instance_inputs(self, idx_instances: list[int]) -> Tensor:
        pass

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        if not hasattr(self, '_dataset'):
            self._open_hdf5()

        # Get bag and instance index
        bag_name = self.spot_names[idx]
        idx_instances = self.mapping_dict[bag_name]

        # Get inputs
        ins_input = self.get_instance_inputs(idx_instances)

        # Get label information (bag + ins)
        bag_labels = self.bag_labels[idx]
        bag_library = self.library_size[idx]
        bag_batch = self.batch_idx[idx]
        bag_id = self.spot_idx[idx]
        instance_labels = self.instance_labels[idx_instances]

        return {
            REGISTRY_KEYS.X_KEY: ins_input,
            REGISTRY_KEYS.Y_BAG_KEY: bag_labels,
            REGISTRY_KEYS.LIBRARY_KEY: bag_library,
            REGISTRY_KEYS.Y_INS_KEY: instance_labels,
            REGISTRY_KEYS.BAG_IDX_KEY: bag_name,
            REGISTRY_KEYS.INS_IDX_KEY: torch.as_tensor(idx_instances),
            REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY: bag_id
            * torch.ones(instance_labels.size(0), dtype=int),
            REGISTRY_KEYS.BATCH_BAG_IDX_KEY: bag_batch,
            REGISTRY_KEYS.BATCH_INS_IDX_KEY: bag_batch
            * torch.ones(instance_labels.size(0), dtype=int),
        }


class EmbeddedMilDataset(BaseDataset):
    h5_key: str = "embeddings"

    def __init__(
        self,
        adata: AnnData,
        instance_folder: str,
    ) -> None:
        super().__init__(
            adata,
            instance_folder,
        )
        self.mean = None
        self.std = None

    def get_instance_inputs(self, idx_instances: list[int]) -> Tensor:
        emb = torch.stack([torch.from_numpy(self._dataset[idx]) for idx in idx_instances]).float()
        if self.mean is not None:
            return (emb - self.mean) / (self.std)
        else:
            return emb


class CelltypeMilDataset(BaseDataset):
    def get_instance_inputs(self, idx_instances: list[int]) -> Tensor:
        return torch.nn.functional.one_hot(
            self.instance_labels[idx_instances].squeeze(1), self.n_class_instance
        ).float()

    def _open_hdf5(self):
        pass


class BaseInstanceDataset(Dataset, ABC):
    h5_key: str

    def __init__(
        self,
        instance_labels: ndarray,
        instance_folder: str,
    ) -> None:
        super().__init__()
        self.instance_folder = instance_folder
        # Prepare instance labels
        unique_values, inverse = np.unique(instance_labels, return_inverse=True)
        self.n_class_instance = len(unique_values)
        self.instance_labels = torch.from_numpy(inverse)
        self.instance_class = instance_labels
        if len(self.instance_labels.shape) == 1:
            self.instance_labels = torch.unsqueeze(self.instance_labels, dim=1)

    def _open_hdf5(self):
        self._h5file = h5py.File(self.instance_folder, 'r')
        self._dataset = self._h5file[self.h5_key]

    def __len__(self) -> int:
        return len(self.instance_labels)

    @abstractmethod
    def get_instance_inputs(self, idx_instances: int) -> Tensor:
        pass

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        if not hasattr(self, '_dataset'):
            self._open_hdf5()

        # Get instance inputs
        ins = self.get_instance_inputs(idx)

        # Get label information
        cell_labels = self.instance_labels[idx]

        return {
            REGISTRY_KEYS.X_KEY: ins,
            REGISTRY_KEYS.Y_INS_KEY: cell_labels,
            REGISTRY_KEYS.INS_IDX_KEY: idx,
        }


class EmbeddedInstanceDataset(BaseInstanceDataset):
    h5_key: str = "embeddings"

    def __init__(
        self,
        instance_labels: ndarray,
        instance_folder: str,
    ):
        super().__init__(instance_labels, instance_folder)
        self.mean = None
        self.std = None

    def get_instance_inputs(self, idx_instance: int) -> Tensor:
        emb = torch.from_numpy(self._dataset[idx_instance]).float()
        if self.mean is not None:
            return (emb - self.mean) / (self.std)
        else:
            return emb


class CelltypeInstanceDataset(BaseInstanceDataset):
    def get_instance_inputs(self, idx_instance: int) -> Tensor:
        return torch.nn.functional.one_hot(
            self.instance_labels[idx_instance].squeeze(), self.n_class_instance
        ).float()

    def _open_hdf5(self):
        pass
