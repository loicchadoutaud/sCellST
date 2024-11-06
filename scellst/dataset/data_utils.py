import logging
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from numpy import ndarray
from sklearn.model_selection import StratifiedKFold

from scellst.config import PreprocessingConfig

logger = logging.getLogger(__name__)


def load_anndata(data_folder: str) -> AnnData:
    """
    Load anndata from h5ad file.

    Args:
        data_folder: Path to the h5ad file.
    """
    # Load AnnData
    adata = sc.read_h5ad(data_folder)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    adata = adata[:, adata.var_names.sort_values().values].copy()

    # Add slide information
    adata.obs["slide_idx"] = 0
    adata.obs["spot_idx"] = np.arange(len(adata))
    return adata


def preprocess_adata(
    adata: AnnData, normalize: bool = True, log1p: bool = True, filtering: bool = True
) -> AnnData:
    # Filter genes that are not expressed
    if filtering:
        logger.info(
            f"Filtering genes based on number of counts: n_genes before filtering: {adata.shape[1]}"
        )
        sc.pp.filter_genes(adata, min_counts=200)
        sc.pp.filter_genes(adata, min_cells=adata.shape[0] // 10)
        sc.pp.filter_cells(adata, min_counts=20)
        logger.info(f"After count filtering: {adata.shape[1]}")
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        adata = adata[:, ~(adata.var["mt"])].copy()
        logger.info(f"After mt and rps filtering: {adata.shape[1]}")

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Preprocess data
    adata.layers["counts"] = adata.X.copy()
    if normalize:
        logger.info(f"Normalising spot counts.")
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log1p:
        logger.info(f"Log1p transform counts.")
        sc.pp.log1p(adata)

    # Compute library size with HVG
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=2000,
        layer="counts",
        flavor="seurat_v3",
        subset=False,
    )
    return adata


def prepare_mil_adata(adata: AnnData, config: PreprocessingConfig) -> AnnData:
    # Subset only relevant cells
    all_spot_cell_map = adata.uns["MIL"]["spot_cell_map"]
    all_spot_cell_distance = adata.uns["MIL"]["spot_cell_distance"]
    new_spot_cell_map, new_spot_cell_distance = {}, {}
    for spot_name in all_spot_cell_map.keys():
        kept_idxs = np.argwhere(
            all_spot_cell_distance[spot_name] <= config.radius_ratio
        ).squeeze(1)
        if len(kept_idxs) > 0:
            new_spot_cell_map[spot_name] = [
                all_spot_cell_map[spot_name][i] for i in kept_idxs
            ]
            new_spot_cell_distance[spot_name] = [
                all_spot_cell_distance[spot_name][i] for i in kept_idxs
            ]

    # Subset anndata to non-empty spots
    adata = adata[list(new_spot_cell_map.keys())]
    adata.uns["MIL"]["spot_cell_map"] = new_spot_cell_map
    adata.uns["MIL"]["spot_cell_distance"] = new_spot_cell_distance

    # All cells in spot
    adata.uns["MIL"]["cell_in_spot"] = list(
        set(
            np.concatenate(
                [
                    cell_indexes
                    for cell_indexes in adata.uns["MIL"]["spot_cell_map"].values()
                ]
            )
        )
    )
    adata.uns["MIL"]["cell_out_spot"] = list(
        set(np.arange(len(adata.uns["MIL"]["cell_label"]))).difference(
            adata.uns["MIL"]["cell_in_spot"]
        )
    )

    return adata


def load_anndatas(list_data_folder: list[str]) -> dict[str, AnnData]:
    return {
        data_folder.split("/")[-1]: load_anndata(
            os.path.join(data_folder, "mil/adata_with_mil.h5ad")
        )
        for data_folder in list_data_folder
    }


def preprocess_anndatas(
    dict_adata: dict[str, AnnData], config: PreprocessingConfig
) -> dict[str, AnnData]:
    return {
        key: preprocess_adata(adata, config.normalize, config.log1p, config.filtering)
        for key, adata in dict_adata.items()
    }


def prepare_mil_anndatas(
    dict_adata: dict[str, AnnData], config: PreprocessingConfig
) -> dict[str, AnnData]:
    output_dict_adata = {}
    for i, (key, adata) in enumerate(dict_adata.items()):
        adata = prepare_mil_adata(adata, config)
        adata.obs["slide_idx"] = i
        output_dict_adata[key] = adata
    return output_dict_adata


def get_common_genes(genes: list[str], dict_adata: dict[str, AnnData]) -> list[str]:
    return sorted(
        list(
            set(genes).intersection(
                *[adata.var_names.tolist() for adata in dict_adata.values()]
            )
        )
    )


def select_HVG(adata: AnnData, n_genes: int) -> list[str]:
    logger.info("Selecting HVG genes...")
    # list_genes = []
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_genes,
        layer="counts",
        flavor="seurat_v3",
        batch_key="slide_idx" if "slide_idx" in adata.obs_names else None,
        subset=False,
    )
    return adata.var.index[adata.var["highly_variable"]]


def select_custom_gene_list(adata: AnnData, file_name: str) -> list[str]:
    file_path = os.path.join("data", file_name)
    if file_name.endswith(".csv"):
        markers = pd.read_csv(file_path, index_col=0)["names"].tolist()
    elif file_name.endswith(".pkl"):
        with open(file_path, "rb") as f:
            markers = pickle.load(f)
    elif file_name.endswith(".txt"):
        markers = pd.read_csv(file_path, sep=" ", header=None)[0].tolist()
    else:
        raise ValueError(f"File name {file_name} not recognised. ")
    logger.info(f"Found {len(markers)} marker genes in database.")
    common_markers = list(set(adata.var_names).intersection(markers))
    logger.info(f"Found {len(common_markers)} / {len(markers)} common genes in adata.")
    return common_markers


def select_labels(adata: AnnData, config: PreprocessingConfig) -> AnnData:
    if len(config.gene_to_pred) != 0:
        list_genes = config.gene_to_pred
        list_genes = [f for f in list_genes if f in adata.var.index]
        list_genes = list(set(list_genes))
        logger.info(
            f"Kept {len(list_genes)} / {len(config.gene_to_pred)} for analysis."
        )
    else:
        if config.gene_type == "HVG":
            list_genes = select_HVG(adata, config.n_genes)
        elif config.gene_type.endswith((".pkl", ".csv", ".txt")):
            list_genes = select_custom_gene_list(adata, config.gene_type)
        else:
            raise ValueError(
                f"Parameter gene type must be one of HVG, HMG, SVG, HMVG, PanglaoDB \nGot {config.gene_type}"
            )
    config.gene_to_pred = sorted(list_genes)
    return adata[:, config.gene_to_pred].copy()


def select_labels_multiple_slides(
    dict_adata: dict[str, AnnData], config: PreprocessingConfig
) -> dict[str, AnnData]:
    # Update config with gene to pred
    full_adata = ad.concat(dict_adata.values())
    _ = select_labels(full_adata, config)

    output_dict_adata = {}
    for key, adata in dict_adata.items():
        output_dict_adata[key] = select_labels(adata, config)
    return output_dict_adata


def split_train_val(y: ndarray, n_folds: int) -> tuple[ndarray, ndarray]:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)
    train_idx, val_idx = next(splitter.split(np.ones_like(y), y))
    return train_idx, val_idx


def split_train_val_all(adata: AnnData, n_folds: int) -> dict[str, ndarray]:
    """Function to split data into train, validation and test sets for different splits."""
    split_dict = {}
    train_idxs, val_idxs = split_train_val(adata.obs["slide_idx"].values, n_folds)
    split_dict["train"] = train_idxs
    split_dict["val"] = val_idxs
    return split_dict
