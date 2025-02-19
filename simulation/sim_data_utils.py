from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from torch.utils.data import Dataset
from scipy import sparse
from scellst.constant import REGISTRY_KEYS
from scellst.dataset.data_handler import VisiumHandler


class InstanceDataset(Dataset):
    h5_key: str = "embedding"

    def __init__(
        self,
        adata: AnnData,
        data_dir: Path,
    ) -> None:
        super().__init__()

        self.cell_embedding_path = adata.uns["cell_embedding_path"]
        assert (
            self.cell_embedding_path.exists()
        ), f"Cell embedding file not found: {self.cell_embedding_path}"
        if isinstance(adata.X, sparse.csr_matrix):
            self.labels = adata.X.toarray()
        else:
            self.labels = adata.X
        self.labels = torch.from_numpy(self.labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def _open_hdf5(self):
        self._h5file = h5py.File(self.cell_embedding_path, "r", swmr=True)
        assert (
            self.h5_key in self._h5file
        ), f"{self.h5_key} not found in h5 file (keys: {self._h5file.keys()})"
        self._dataset = self._h5file[self.h5_key]
        self._instance_labels = self._h5file["label"]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        if not hasattr(self, "_dataset"):
            self._open_hdf5()
        return {
            REGISTRY_KEYS.X_KEY: torch.from_numpy(self._dataset[idx]).float(),
            REGISTRY_KEYS.Y_INS_KEY: self.labels[idx],
        }


class InstanceHandler(VisiumHandler):
    def create_adata(self, embedding_path: Path, id: str) -> AnnData:
        assert embedding_path.exists(), f"File {embedding_path} does not exist."
        h5_file = h5py.File(embedding_path, mode="r")
        expression = h5_file["expression"][:]
        barcode = h5_file["barcode"][:]
        gene_names = h5_file["gene_names"][:]
        adata = AnnData(
            X=expression,
            obs=pd.DataFrame(index=barcode),
            var=pd.DataFrame(index=gene_names),
        )
        adata.uns["hest_id"] = id
        logger.info(f"Loaded adata with shape: {adata.shape}")
        return adata

    def load_and_preprocess_data(
        self,
        data_dir: Path,
        id: str,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
        cell_embedding_path: Path,
    ) -> AnnData:
        logger.info(f"Loading data for ID: {id}")
        adata = self.create_adata(cell_embedding_path, id)
        adata = self.preprocess_data(
            adata, filter_genes, filter_cells, normalize, log1p
        )
        adata.uns["cell_embedding_path"] = cell_embedding_path
        logger.info("Preprocessing completed.")
        return adata

    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return InstanceDataset(adata, data_dir)


class SupervisedInstanceHandler(VisiumHandler):
    def create_adata(self, embedding_path: Path, id: str) -> AnnData:
        assert embedding_path.exists(), f"File {embedding_path} does not exist."
        h5_file = h5py.File(embedding_path, mode="r")
        expression = h5_file["expression"][:]
        barcode = h5_file["barcode"][:]
        gene_names = h5_file["gene_names"][:]
        adata = AnnData(
            X=expression,
            obs=pd.DataFrame(index=barcode),
            var=pd.DataFrame(index=gene_names),
        )
        adata.uns["hest_id"] = id
        logger.info(f"Loaded adata with shape: {adata.shape}")
        return adata

    def load_and_preprocess_data(
        self,
        data_dir: Path,
        id: str,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
        cell_embedding_path: Path,
    ) -> AnnData:
        logger.info(f"Loading data for ID: {id}")
        adata = self.create_adata(cell_embedding_path, id)
        adata = self.preprocess_data(
            adata, filter_genes, filter_cells, normalize, log1p
        )
        adata.uns["cell_embedding_path"] = cell_embedding_path
        logger.info("Preprocessing completed.")
        return adata

    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return InstanceDataset(adata, data_dir)
