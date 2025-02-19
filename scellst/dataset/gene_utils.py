import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from anndata import AnnData
from loguru import logger
from omegaconf import ListConfig
from pandas import DataFrame
from torch import Tensor

from scellst.constant import CLASS_LABELS, DATA_DIR


def select_labels(adata: AnnData, genes: str | list[str]) -> list[str]:
    if isinstance(genes, str):
        if (DATA_DIR / f"genes_{genes}.csv").exists():
            file_name = f"genes_{genes}.csv"
            logger.info(f"Loading genes from file: {file_name}")
            df = pd.read_csv(DATA_DIR / file_name)
            genes_to_pred = [f for f in df["gene"] if f in adata.var_names]
            genes_to_pred = list(set(genes_to_pred))
            logger.info(
                f"Found {len(genes_to_pred)} / {len(df)} genes in adata.var_names"
            )
        else:
            logger.info(f"Selecting {genes}.")
            gene_type = genes.split(":")[0]
            n_genes = int(genes.split(":")[1])
            if gene_type == "HVG":
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=n_genes,
                    subset=False,
                    layer="counts",
                    flavor="seurat_v3",
                )
                genes_to_pred = adata.var_names[adata.var["highly_variable"]].tolist()
            elif gene_type == "SVG":
                sq.gr.spatial_neighbors(adata, n_rings=1, coord_type="grid", n_neighs=6)
                sq.gr.spatial_autocorr(
                    adata,
                    mode="moran",
                )
                genes_to_pred = adata.uns["moranI"]["I"].index[:n_genes].tolist()
            else:
                raise ValueError(f"Unknown gene type: {gene_type}")
    else:
        logger.info(f"Selecting genes from list.")
        assert isinstance(
            genes, ListConfig
        ), f"Expected list of genes, got {type(genes)}"
        genes_to_pred = [g for g in genes if g in adata.var_names]
        genes_to_pred = list(set(genes_to_pred))
        logger.info(
            f"Found {len(genes_to_pred)} / {len(genes)} genes in adata.var_names"
        )
    genes_to_pred.sort()
    if len(genes_to_pred) == 0:
        raise ValueError("No genes found in adata.var_names.")
    return genes_to_pred


def convert_signature_to_mask(df: DataFrame, gene_to_pred: list[str]) -> Tensor:
    """
    Convert a signature Dataframe to a mask.
    The dataframe is expected to have the following columns:
    - gene: gene name
    - celltype: celltype name
    - celltype_idx: index of the celltype

    The output is a mask of shape (n_celltypes, n_genes) where each row corresponds to a celltype and each column to a gene ordered by alphabetical order.
    A value of 1 indicates that the gene is expressed in the celltype.
    """
    df = df.sort_values(by="gene")
    df.set_index("gene", inplace=True)
    mask = torch.zeros(
        (len(CLASS_LABELS.keys()), len(gene_to_pred)), dtype=torch.float32
    )
    for i, gene in enumerate(gene_to_pred):
        if gene in df.index:
            celltype = df.loc[gene]["celltype"]
            mask[CLASS_LABELS[celltype], i] = 1
    # Set column for unknown celltype to 1
    mask[:, mask.sum(dim=0) == 0] = 1
    return mask


def create_mask_prior(signature_path: str, gene_to_pred: list[str]) -> Tensor:
    """
    Create a mask prior from a signature file.
    """
    df = pd.read_csv(signature_path)
    return convert_signature_to_mask(df, gene_to_pred)
