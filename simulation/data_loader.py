from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
import scanpy as sc


class SimulationDataLoader:
    def __init__(self, data_path: Path, embedding_key: str, ref_adata_path: Path):
        embedding_path = data_path / "cell_embeddings" / f"{embedding_key}.h5"
        assert (
            embedding_path.exists()
        ), f"Embedding path does not exist, got: {embedding_path}"
        assert (
            ref_adata_path.exists()
        ), f"Reference AnnData path does not exist, got: {ref_adata_path}"
        self.embedding_path = embedding_path
        self.ref_adata_path = ref_adata_path

    def load_embedding(self) -> AnnData:
        h5_file = h5py.File(self.embedding_path, "r")
        embeddings = h5_file["embedding"][:]
        labels = h5_file["label"][:].squeeze()
        adata = AnnData(
            X=embeddings,
            obs=pd.DataFrame({"id": np.arange(len(embeddings)), "class": labels}),
        )
        data_info = self.embedding_path.stem.split("_")
        adata.obs["slide"] = data_info[1]
        adata.uns["embedding_key"] = data_info[0]
        adata.uns["sample_id"] = data_info[1]
        adata.uns["tag"] = data_info
        logger.info(f"Loaded embedding data with shape {adata.X.shape}")
        return adata

    def load_ref_adata(self, subset_col: str, value_to_keep: list[str]) -> AnnData:
        # Load data
        adata = sc.read_h5ad(self.ref_adata_path, backed="r")
        adata = adata[
            adata.obs[subset_col].isin(value_to_keep)
        ].to_memory()  # Subset only relevant cells
        adata = adata.raw.to_adata()
        adata.var["feature_name"] = adata.var["feature_name"].astype(str)
        adata.var.set_index("feature_name", inplace=True)
        adata.var_names_make_unique()
        logger.info(f"Loaded reference AnnData with shape {adata.X.shape}")
        return adata

    def load_data(
        self, subset_col: str, value_to_keep: list[str]
    ) -> tuple[AnnData, AnnData]:
        return self.load_embedding(), self.load_ref_adata(subset_col, value_to_keep)
