import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from anndata import AnnData
from scipy import sparse
from torch import Tensor
from torch.utils.data import Dataset

from scellst.constant import REGISTRY_KEYS


class EmbeddedMilDataset(Dataset):
    h5_key: str = "embedding"

    def __init__(
        self,
        adata: AnnData,
        data_dir: Path,
    ) -> None:
        super().__init__()
        self.cell_embedding_path = adata.uns["cell_embedding_path"]
        assert os.path.exists(
            self.cell_embedding_path
        ), f"Cell embedding file not found: {self.cell_embedding_path}"
        self.ser_map = adata.uns["spot_cell_map"]
        self.spot_names = adata.obs_names
        if isinstance(adata.X, sparse.csr_matrix):
            self.bag_labels = adata.X.toarray()
        else:
            self.bag_labels = adata.X
        self.bag_labels = torch.from_numpy(self.bag_labels).float()
        self.size_factor = adata.obs["size_factor"].values

    def __len__(self) -> int:
        return len(self.bag_labels)

    def _open_hdf5(self):
        self._h5file = h5py.File(self.cell_embedding_path, "r", swmr=True)
        assert (
            self.h5_key in self._h5file
        ), f"{self.h5_key} not found in h5 file (keys: {self._h5file.keys()})"
        self._dataset = self._h5file[self.h5_key]
        self._instance_labels = self._h5file["label"]

    def get_instance_inputs(self, idx_instances: list[int]) -> Tensor:
        emb = torch.stack(
            [torch.from_numpy(self._dataset[idx]) for idx in idx_instances]
        ).float()
        return emb

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        if not hasattr(self, "_dataset"):
            self._open_hdf5()

        # Get bag and instance index
        bag_name = self.spot_names[idx]
        idx_instances = self.ser_map[bag_name]

        # Get inputs
        ins_input = self.get_instance_inputs(idx_instances)

        # Get label information
        bag_labels = self.bag_labels[idx]
        instance_labels = torch.from_numpy(
            self._instance_labels[idx_instances]
        ).squeeze(1)

        # Get size factor
        size_factor = self.size_factor[idx]

        # Convert unique bag_name to int
        bag_idx = string_to_int(bag_name)

        return {
            REGISTRY_KEYS.X_KEY: ins_input,
            REGISTRY_KEYS.Y_BAG_KEY: bag_labels,
            REGISTRY_KEYS.Y_INS_KEY: instance_labels,
            REGISTRY_KEYS.INSTANCE_BAG_IDX_KEY: torch.repeat_interleave(
                torch.tensor([bag_idx]), len(idx_instances)
            ),
            REGISTRY_KEYS.SIZE_FACTOR: size_factor,
        }


class EmbeddedInstanceDataset(Dataset):
    h5_key: str = "embedding"

    def __init__(
        self, cell_embedding_path: Path, idx_to_pred: np.ndarray | None = None
    ) -> None:
        super().__init__()
        self.cell_embedding_path = cell_embedding_path
        assert (
            self.cell_embedding_path.exists()
        ), f"Cell embedding file not found: {self.cell_embedding_path}"

        if idx_to_pred is None:
            with h5py.File(self.cell_embedding_path, "r", swmr=True) as h5file:
                idx_to_pred = np.arange(
                    len(h5file["barcode"][:].flatten().astype(str).astype("object"))
                )
        self.n_inst = idx_to_pred.shape[0]
        self.idx_to_pred = idx_to_pred

    def __len__(self) -> int:
        return self.n_inst

    def _open_hdf5(self):
        self._h5file = h5py.File(self.cell_embedding_path, "r", swmr=True)
        self._dataset = self._h5file[self.h5_key]
        assert (
            self.h5_key in self._h5file.keys()
        ), f"{self.h5_key} not found in h5 file (keys: {self._h5file.keys()})"

    def get_instance_inputs(self, idx_instance: int) -> Tensor:
        return torch.from_numpy(self._dataset[idx_instance]).float()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        if not hasattr(self, "_dataset"):
            self._open_hdf5()

        return {
            REGISTRY_KEYS.X_KEY: self.get_instance_inputs(self.idx_to_pred[idx]),
        }


def string_to_int(string: str) -> int:
    return int(hashlib.md5(string.encode()).hexdigest(), 16) % (10**8)
