from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from loguru import logger

from scellst.constant import DATA_DIR
from scellst.dataset.data_handler import VisiumHandler


def compute_common_svg(adata: ad.AnnData, n_top_genes: int, batch_key: str) -> AnnData:
    df = pd.DataFrame(index=adata.var_names)
    for batch in adata.obs[batch_key].unique():
        adata_batch = adata[adata.obs[batch_key] == batch].copy()
        sq.gr.spatial_neighbors(adata_batch, n_rings=1, coord_type="grid", n_neighs=6)
        sq.gr.spatial_autocorr(
            adata_batch,
            mode="moran",
        )
        df.join(adata_batch.uns["moranI"]["I"].rename(batch))

    # Compute mean score
    df["mean"] = df.apply(lambda row: np.exp(np.log(row).mean()), axis=1)
    df = df.sort_values("mean", ascending=False)
    return adata[:, df.index[:n_top_genes].tolist()].copy()


def compute_common_heg(adata: ad.AnnData, n_top_genes: int, batch_key: str) -> AnnData:
    df = pd.DataFrame(index=adata.var_names)
    for batch in adata.obs[batch_key].unique():
        adata_batch = adata[adata.obs[batch_key] == batch].copy()
        df.join(adata_batch.X.mean(axis=0).rename(batch))

    # Compute mean score
    df["mean"] = df.mean(axis=1)
    df = df.sort_values("mean", ascending=False)
    return adata[:, df.index[:n_top_genes].tolist()].copy()


def compute_common_hvg(
    adata: ad.AnnData, n_top_genes: int, batch_key: str
) -> ad.AnnData:
    """
    Compute common highly variable genes across batches and select top n_top_genes based on
    the number of times a gene is selected as HVG.

    Parameters:
        adata: AnnData
            Input AnnData object with `batch_key` in `adata.obs`.
        n_top_genes: int
            Number of top genes to select.
        batch_key: str
            Column in `adata.obs` indicating batch labels.

    Returns:
        ad.AnnData: A new AnnData object with only the selected HVG genes.
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="batch",
    )

    # Create a new AnnData with only the selected HVGs
    return adata


def prepare_list_hvg(
    data_dir: Path, list_ids: list[str], n_genes: int, organ: str
) -> None:
    # Load and get hvg genes
    list_adata = []
    for id in list_ids:
        list_adata.append(
            VisiumHandler().load_and_preprocess_data(
                data_dir,
                id,
                filter_genes=True,
                filter_cells=True,
                normalize=True,
                log1p=True,
                embedding_path=None,
            )
        )
    adata = ad.concat(list_adata, join="inner", index_unique="_", label="batch")

    # Compute HVG
    adata = compute_common_hvg(adata, n_top_genes=n_genes, batch_key="batch")

    # Save results
    pd.Series(adata.var_names, name="gene").sort_values().to_csv(
        DATA_DIR / f"genes_{organ}_{n_genes}_hvg_bench.csv"
    )
    logger.info("End of gene selection without errors.")


def prepare_list_svg(
    data_dir: Path, list_ids: list[str], n_genes: int, organ: str
) -> None:
    list_adata = []
    for id in list_ids:
        list_adata.append(
            VisiumHandler().load_and_preprocess_data(
                data_dir,
                id,
                filter_genes=True,
                filter_cells=True,
                normalize=True,
                log1p=True,
                embedding_path=None,
            )
        )
    adata = ad.concat(list_adata, join="inner", index_unique="_", label="batch")

    # Compute SVG
    adata = compute_common_svg(adata, n_top_genes=n_genes, batch_key="batch")

    # Save results
    pd.Series(adata.var_names, name="gene").sort_values().to_csv(
        DATA_DIR / f"genes_{organ}_{n_genes}_svg_bench.csv"
    )
    logger.info("End of gene selection without errors.")


def prepare_list_heg(
    data_dir: Path, list_ids: list[str], n_genes: int, organ: str
) -> None:
    list_adata = []
    for id in list_ids:
        list_adata.append(
            VisiumHandler().load_and_preprocess_data(
                data_dir,
                id,
                filter_genes=True,
                filter_cells=True,
                normalize=True,
                log1p=True,
                embedding_path=None,
            )
        )
    adata = ad.concat(list_adata, join="inner", index_unique="_", label="batch")

    # Compute SVG
    adata = compute_common_svg(adata, n_top_genes=n_genes, batch_key="batch")

    # Save results
    pd.Series(adata.var_names, name="gene").sort_values().to_csv(
        DATA_DIR / f"genes_{organ}_{n_genes}_heg_bench.csv"
    )
    logger.info("End of gene selection without errors.")
