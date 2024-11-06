import argparse
import os.path

import numpy as np
import scanpy as sc
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

from anndata import AnnData
from numpy import ndarray
from scellst.logger import logger
from scellst.eval_utils import eval_reg_metrics


def eval_predictions(adata: AnnData, output_path: str, suffix: str) -> None:
    logger.info(adata.shape)
    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    if ("nb" in adata.uns["exp_id"]) and (suffix == "spot"):
        logger.info("Removing effect of library size")
        adata.X /= adata.obs["total_counts"].values[:, np.newaxis]
        adata.layers["predictions"] /= adata.obs["total_counts"].values[:, np.newaxis]
    df_metrics = eval_reg_metrics(
        adata.layers["predictions"],
        adata.X,
        adata.var_names.values,
        suffix
    )
    df_metrics["tag"] = adata.uns["exp_id"]
    os.makedirs(output_path, exist_ok=True)
    df_metrics.to_csv(os.path.join(output_path, adata.uns["exp_id"] + ".csv"))


def compare_correlation_structure(X_pred: ndarray, X_true: ndarray, output_path: str, tag: str) -> None:
    for X, name in zip([X_pred, X_true], ["pred", "true"]):
        corr = np.corrcoef(X, rowvar=False)
        sns.clustermap(corr, vmin=-1, vmax=1, row_cluster=False, col_cluster=False)
        plt.savefig(os.path.join(output_path, tag + f"_{name}_correlation.png"))
        plt.close()


def eval_all_spot_predictions(prediction_folder: str, exp_tag: str) -> None:
    list_anndata_path = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith(".h5ad")]
    list_anndata_path.sort()
    output_path = f"outputs/metrics/{exp_tag}/spot"
    os.makedirs(output_path, exist_ok=True)
    for filename in list_anndata_path:
        logger.info(os.path.basename(filename))
        adata = sc.read_h5ad(filename)
        adata.uns["exp_id"] = os.path.splitext(os.path.basename(filename))[0]
        eval_predictions(adata, output_path, "spot")


def load_cell_predictions_with_labels(filename: str, data_folder: str) -> AnnData:
    cell_adata = sc.read_h5ad(filename)
    label_filename = os.path.join(data_folder, cell_adata.uns['slide_name'], "cell_adata_labels.h5ad")
    cell_labels = sc.read_h5ad(label_filename)
    assert (cell_labels.obs_names == cell_adata.obs_names).all(), "different cell index found."
    common_genes = list(set(cell_adata.var_names).intersection(cell_labels.var_names))
    common_genes.sort()
    cell_adata = cell_adata[:, common_genes]
    cell_labels = cell_labels[:, common_genes]
    cell_labels.layers["predictions"] = cell_adata.X
    cell_labels.uns["exp_id"] = cell_adata.uns["exp_id"]
    logger.info(cell_labels.shape)
    logger.info("Normalizing count labels for cells.")
    sc.pp.normalize_total(cell_labels)
    return cell_labels


def eval_all_cell_predictions(prediction_folder: str, exp_tag: str, data_folder: str) -> None:
    logger.info(os.listdir(prediction_folder))
    list_anndata_path = [os.path.join(prediction_folder, f) for f in os.listdir(prediction_folder) if f.endswith(".h5ad")]
    list_anndata_path.sort()
    output_path = f"outputs/metrics/{exp_tag}/cell"
    os.makedirs(output_path, exist_ok=True)
    for filename in list_anndata_path:
        logger.info(os.path.basename(filename))
        adata = load_cell_predictions_with_labels(filename, data_folder)
        adata.uns["exp_id"] = os.path.splitext(os.path.basename(filename))[0]
        eval_predictions(adata, output_path, "cell")


if __name__ == '__main__':
    # Read arguments
    parser = argparse.ArgumentParser(description="Mil despot")
    parser.add_argument(
        "--prediction_spot_folder",
        default=None,
        type=str,
        help="file path (default: None)",
    )
    parser.add_argument(
        "--prediction_cell_folder",
        default=None,
        type=str,
        help="file path (default: None)",
    )
    parser.add_argument(
        "--data_folder",
        default=None,
        type=str,
        help="file path (default: None)",
    )
    parser.add_argument(
        "--exp_tag",
        default=None,
        type=str,
        help="file path (default: None)",
    )
    args = parser.parse_args()
    if args.prediction_spot_folder is not None:
        eval_all_spot_predictions(args.prediction_spot_folder, args.exp_tag)
    if args.prediction_cell_folder is not None:
        eval_all_cell_predictions(args.prediction_cell_folder, args.exp_tag, args.data_folder)
    logger.info("End of python script.")