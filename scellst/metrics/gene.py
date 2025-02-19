import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from numpy import ndarray

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


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

    # Pearson correlation for each target
    pcc = [pearsonr(Y[:, i], Y_pred[:, i])[0] for i in range(Y.shape[1])]

    # Spearman correlation for each target
    scc = [spearmanr(Y[:, i], Y_pred[:, i])[0] for i in range(Y.shape[1])]

    # MSE for each target
    mse = [mean_squared_error(Y[:, i], Y_pred[:, i]) for i in range(Y.shape[1])]

    # Combine metrics into a dictionary
    metrics = {f"pcc/{target}": pcc_val for target, pcc_val in zip(target_names, pcc)}
    metrics.update(
        {f"scc/{target}": scc_val for target, scc_val in zip(target_names, scc)}
    )
    metrics.update(
        {f"mse/{target}": mse_val for target, mse_val in zip(target_names, mse)}
    )

    # Log summary metrics
    logger.info(f"Mean pcc: {np.nanmean(pcc):.2f}")
    logger.info(f"Mean scc: {np.nanmean(scc):.2f}")
    logger.info(f"Mean mse: {np.nanmean(mse):.2f}")

    return metrics


def compute_gene_metrics(adata: AnnData, adata_pred: AnnData) -> pd.DataFrame:
    """Compute both supervised metrics."""
    logger.info("Starting metrics computation.")

    # Find common genes
    predicted_set = set(adata_pred.var_names)
    common_genes = [g for g in adata.var_names if g in predicted_set]
    logger.info(
        f"Found {len(common_genes)} / {len(adata.var_names)} in measured genes."
    )
    logger.info(
        f"Found {len(common_genes)} / {len(adata_pred.var_names)} in predicted genes."
    )
    adata = adata[:, common_genes]
    adata_pred = adata_pred[:, common_genes]

    # Supervised metrics
    logger.info("Starting supervised metrics computation.")
    supervised_metrics = compute_regression_metrics(adata.X, adata_pred.X, common_genes)
    logger.info("Metrics computed.")

    return pd.DataFrame([supervised_metrics])
