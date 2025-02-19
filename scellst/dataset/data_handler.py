from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import scanpy as sc
import torch
from anndata import AnnData
from loguru import logger

from scellst.cellhest_adapter.cell_utils import create_spot_cell_map
from scellst.dataset.torch_dataset import EmbeddedInstanceDataset, EmbeddedMilDataset


class BaseHandler(ABC):
    def load_data(self, data_dir: Path, id: str) -> AnnData:
        adata_path = self.get_adata_path(data_dir, id)
        assert adata_path.exists(), f"File {adata_path} does not exist."
        adata = sc.read_h5ad(adata_path)
        adata.obs_names = [name + f"_{id}" for name in adata.obs_names]
        adata.uns["hest_id"] = id
        logger.info(f"Loaded adata with shape: {adata.shape}")
        return adata

    def preprocess_data(
        self,
        adata: AnnData,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
    ) -> AnnData:
        if filter_genes:
            adata = self.filter_genes(adata)
        if filter_cells:
            adata = self.filter_cells(adata)

        adata = self.filter_mitochondrial_genes(adata)
        logger.info(f"After mt and rps filtering: {adata.shape[1]}")

        adata.obs["n_counts"] = adata.X.sum(1)
        adata.obs["size_factor"] = (
            adata.obs["n_counts"] / adata.obs["n_counts"].median()
        )

        adata.layers["counts"] = adata.X.copy()
        if normalize:
            logger.info(f"Normalising spot counts.")
            sc.pp.normalize_total(adata, target_sum=1e4)
        if log1p:
            logger.info(f"Log1p transform counts.")
            sc.pp.log1p(adata)

        return adata

    @abstractmethod
    def create_dataset(
        self, adata: AnnData, embedding_path: str
    ) -> torch.utils.data.Dataset:
        pass

    @abstractmethod
    def get_adata_path(self, data_dir: Path, id: str) -> Path:
        pass

    @abstractmethod
    def filter_genes(self, adata: AnnData) -> AnnData:
        pass

    @abstractmethod
    def filter_cells(self, adata: AnnData) -> AnnData:
        pass

    def filter_mitochondrial_genes(self, adata: AnnData) -> AnnData:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        return adata[:, ~(adata.var["mt"])].copy()

    @abstractmethod
    def load_and_preprocess_data(
        self,
        data_dir: Path,
        id: str,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
        embedding_path: Path,
    ) -> AnnData:
        pass


class VisiumHandler(BaseHandler):
    def create_dataset(
        self, adata: AnnData, embedding_path: str
    ) -> torch.utils.data.Dataset:
        pass

    def create_inference_dataset(
        self, embedding_path: Path
    ) -> torch.utils.data.Dataset:
        return EmbeddedInstanceDataset(embedding_path)

    def get_adata_path(self, data_dir: Path, id: str) -> Path:
        return data_dir / "st" / f"{id}.h5ad"

    def filter_genes(self, adata: AnnData) -> AnnData:
        sc.pp.filter_genes(adata, min_counts=200)
        sc.pp.filter_genes(adata, min_cells=adata.shape[0] // 10)
        logger.info(f"After genes filtering: {adata.shape}")
        return adata

    def filter_cells(self, adata: AnnData) -> AnnData:
        sc.pp.filter_cells(adata, min_counts=20)
        logger.info(f"After count filtering: {adata.shape}")
        return adata

    def load_and_preprocess_data(
        self,
        data_dir: Path,
        id: str,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
        embedding_path: Path,
    ) -> AnnData:
        logger.info(f"Loading data for ID: {id}")
        adata = self.load_data(data_dir, id)
        adata = self.preprocess_data(
            adata, filter_genes, filter_cells, normalize, log1p
        )
        logger.info("Preprocessing completed.")
        return adata


class MilVisiumHandler(VisiumHandler):
    def create_spot_cell_map(self, adata: AnnData) -> AnnData:
        # Create spot cell map
        ser_map = create_spot_cell_map(adata.uns["cell_embedding_path"])
        ser_map.index = [f"{idx}_{adata.uns['hest_id']}" for idx in ser_map.index]

        # Prepare bag labels
        spot_names = list(set(ser_map.index).intersection(adata.obs_names))
        spot_names.sort()
        logger.info(
            f"Found {len(spot_names)} / {len(adata.obs_names)} spots in the adata file."
        )
        logger.info(
            f"Found {len(spot_names)} / {len(ser_map)} spots in the cell embedding file."
        )
        ser_map = ser_map.loc[spot_names]

        # Filter out spots that do not have cells
        adata = adata[spot_names].copy()
        adata.uns["spot_cell_map"] = ser_map
        return adata

    def load_and_preprocess_data(
        self,
        data_dir: Path,
        id: str,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
        embedding_path: Path,
    ) -> AnnData:
        logger.info(f"Loading data for ID: {id}")
        adata = self.load_data(data_dir, id)
        adata = self.preprocess_data(
            adata, filter_genes, filter_cells, normalize, log1p
        )
        adata.uns["cell_embedding_path"] = embedding_path
        adata = self.create_spot_cell_map(adata)
        logger.info("Preprocessing completed.")
        return adata

    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return EmbeddedMilDataset(adata, data_dir)


class XeniumHandler(BaseHandler):
    def get_adata_path(self, data_dir: Path, id: str) -> Path:
        return data_dir / "cell_genes" / f"{id}.h5ad"

    def create_dataset(
        self, adata: AnnData, embedding_path: str
    ) -> torch.utils.data.Dataset:
        raise ValueError("Should not be called.")

    def create_inference_dataset(
        self, embedding_path: Path
    ) -> torch.utils.data.Dataset:
        return EmbeddedInstanceDataset(embedding_path)

    def filter_genes(self, adata: AnnData) -> AnnData:
        sc.pp.filter_genes(adata, min_cells=5)
        logger.info(f"After genes filtering: {adata.shape}")
        return adata

    def filter_cells(self, adata: AnnData) -> AnnData:
        return adata

    def load_and_preprocess_data(
        self,
        data_dir: Path,
        id: str,
        filter_genes: bool,
        filter_cells: bool,
        normalize: bool,
        log1p: bool,
        embedding_path: Path,
    ) -> AnnData:
        logger.info(f"Loading data for ID: {id}")
        adata = self.load_data(data_dir, id)
        adata = self.preprocess_data(
            adata, filter_genes, filter_cells, normalize, log1p
        )
        adata.uns["cell_embedding_path"] = embedding_path
        logger.info("Preprocessing completed.")
        return adata
