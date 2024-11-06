import logging
from functools import partial

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from scellst.utils import convert_params_to_nb

logger = logging.getLogger(__name__)


def compute_statistics(y_true: ndarray, y_pred: np.ndarray, func: callable) -> ndarray:
    stats = np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        stats[i] = func(y_true[:, i], y_pred[:, i]).statistic
    return stats


def compute_statistics_with_significance(
    y_true: ndarray, y_pred: np.ndarray, func: callable
) -> tuple[ndarray, ndarray]:
    stats, pvals = np.zeros(y_pred.shape[1]), np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        res = func(
            y_true[:, i],
            y_pred[:, i],
            alternative="greater",
        )
        stats[i] = res.statistic
        pvals[i] = res.pvalue
    return stats, pvals


def compute_statistics_with_significance_normalised(
    y_true: ndarray, y_pred: np.ndarray, func: callable
) -> tuple[ndarray, ndarray]:
    library_size = y_true.sum(axis=1)
    stats, pvals = np.zeros(y_pred.shape[1]), np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        res = func(
            y_true[:, i] / library_size,
            y_pred[:, i] / library_size,
            alternative="greater",
        )
        stats[i] = res.statistic
        pvals[i] = res.pvalue
    return stats, pvals


REG_METRICS = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
}

RANK_METRICS = {
    "pcc": partial(compute_statistics, func=pearsonr),
    "scc": partial(compute_statistics, func=spearmanr),
}

STATS_METRICS = {
    "pcc": partial(compute_statistics_with_significance, func=pearsonr),
    "scc": partial(compute_statistics_with_significance, func=spearmanr),
    "pcc_norm": partial(compute_statistics_with_significance_normalised, func=pearsonr),
    "scc_norm": partial(compute_statistics_with_significance_normalised, func=spearmanr),
}


def _check_input_shapes(
    y_pred: ndarray,
    y_true: ndarray,
    label_names: ndarray,
) -> None:
    # Check input shapes are correct
    assert (
        y_pred.shape[0] == y_true.shape[0]
    ), f"Length of y_pred must be equal to y_true,\nGot {y_pred.shape[0]} and {y_true.shape[0]}"
    assert (
        y_pred.shape[1] == label_names.shape[0]
    ), f"Length of label names must be equal to the number of predictions per samples,\nGot {y_pred.shape[1]} and {label_names.shape[0]}"


def eval_reg_metrics(
    y_pred: ndarray,
    y_true: ndarray,
    label_names: ndarray,
    suffix: str,
) -> DataFrame:
    _check_input_shapes(y_pred, y_true, label_names)

    # Compute tmp_metrics
    metrics = {
        f"{key}_{suffix}": value(y_true, y_pred, multioutput="raw_values")
        for key, value in REG_METRICS.items()
    }
    metrics.update(
        {
            f"{key}_{suffix}": value(y_true, y_pred)
            for key, value in RANK_METRICS.items()
        }
    )
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics["label_name"] = label_names
    return df_metrics


def eval_dist_metrics(
    y_pred: ndarray,
    y_true: ndarray,
    label_names: ndarray,
    suffix: str,
) -> DataFrame:
    dist = convert_params_to_nb(y_pred)
    y_mean_pred = dist.mean.numpy()

    # Compute tmp_metrics
    metrics = {
        f"{key}_{suffix}": value(y_true, y_mean_pred, multioutput="raw_values")
        for key, value in REG_METRICS.items()
    }
    metrics.update(
        {f"nll_{suffix}": -dist.log_prob(torch.from_numpy(y_true).int()).mean(dim=0)}
    )
    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics["label_name"] = label_names
    return df_metrics


def eval_statistical_significance(
    y_pred: ndarray,
    y_true: ndarray,
    label_names: ndarray,
    suffix: str,
) -> DataFrame:
    dist = convert_params_to_nb(y_pred)
    y_mean_pred = dist.mean.numpy()
    metrics = {}

    for key, value in STATS_METRICS.items():
        stats, pvals = value(y_true, y_mean_pred)
        metrics[f"{key}_{suffix}_stats"] = stats
        metrics[f"{key}_{suffix}_pvals"] = pvals

    df_metrics = pd.DataFrame.from_dict(metrics)
    df_metrics["label_name"] = label_names
    return df_metrics


eval_metric_func_dict = {
    "regression": eval_reg_metrics,
    "nb_total_regression": eval_reg_metrics,
    "nb_mean_regression": eval_reg_metrics,
}


def eval_model(predictions: dict, split: str, task: str, tag: str, object_type: str) -> DataFrame:
    # Check tasks are correct
    assert (
        task in eval_metric_func_dict.keys()
    ), f"task must be in {eval_metric_func_dict.keys()},\nGot {task}"

    # Compute metrics
    metrics = eval_metric_func_dict[task](
        predictions[split]["predictions"],
        predictions[split]["labels"],
        predictions[split]["gene_names"],
        object_type,
    )

    # Concat tmp_metrics
    metrics["split"] = split
    metrics["tag"] = tag

    return metrics
