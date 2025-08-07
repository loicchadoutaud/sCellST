import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from numpy import ndarray

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(
    Y: ndarray, Y_pred: ndarray, target_names: list[str]
) -> dict[str, float]:
    """
    Compute regression metrics (Pearson and Spearman correlations) for each target.

    Args:
        Y (ndarray): True values with shape (n_samples, n_targets).
        Y_pred (ndarray): Predicted values with shape (n_samples, n_targets).
        target_names (List[str]): Names of the targets.

    Returns:
        Dict[str, float]: Dictionary of metrics with target names as keys.
    """
    if Y.shape != Y_pred.shape:
        raise ValueError("Shapes of Y and Y_pred must match.")
    if len(target_names) != Y.shape[1]:
        raise ValueError(
            "Number of target names must match the number of targets in Y and Y_pred."
        )

    output_dict = {}
    correlations = {"pcc": pearsonr, "scc": spearmanr}
    for corr_name, corr_func in correlations.items():
        corr = [corr_func(Y[:, i], Y_pred[:, i])[0] for i in range(Y.shape[1])]
        output_dict.update({f"{corr_name}/{target}": val for target, val in zip(target_names, corr)})
        logger.info(f"Mean {corr_name}: {np.nanmean(corr):.2f}")

    metrics = {"rmse": root_mean_squared_error, "mae": mean_absolute_error, "r2": r2_score}
    for metric_name, metric_func in metrics.items():
        met = [metric_func(Y[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]
        output_dict.update({f"{metric_name}/{target}": val for target, val in zip(target_names, met)})
        logger.info(f"Mean {metric_name}: {np.nanmean(met):.2f}")

    return output_dict


def compute_gene_metrics(adata: AnnData, adata_pred: AnnData) -> pd.DataFrame:
    """Compute both supervised metrics."""
    logger.info("Starting metrics computation.")

    # Find common genes
    common_genes = list(set(adata.var_names) & set(adata_pred.var_names))
    logger.info(
        f"Found {len(common_genes)} / {len(adata.var_names)} in measured genes."
    )
    logger.info(
        f"Found {len(common_genes)} / {len(adata_pred.var_names)} in predicted genes."
    )
    common_genes = np.sort(common_genes)
    adata = adata[:, common_genes]
    adata_pred = adata_pred[:, common_genes]

    logger.info(f"Computing supervised metrics on {adata.shape} and {adata_pred.shape}.")

    # Supervised metrics
    logger.info("Starting supervised metrics computation.")
    supervised_metrics = compute_regression_metrics(adata.X, adata_pred.X, common_genes)
    logger.info("Metrics computed.")

    return pd.DataFrame([supervised_metrics])
