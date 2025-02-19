from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from lightning import Trainer
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from scellst.constant import METRICS_DIR, FIGURES_DIR, PREDS_DIR, REV_CLASS_LABELS
from scellst.dataset.data_module import prepare_data_module, STDataModule
from scellst.io_utils import load_yaml
from scellst.lightning_model.base_lightning_model import BaseLightningModel
from scellst.metrics.gene import compute_gene_metrics
from scellst.metrics.metric_utils import format_metric_df
from scellst.plots.plot_spatial import plot_top_genes
from scellst.utils import update_config, load_model


def format_predictions(
    predictions: list[Tensor], data_module: STDataModule, infer_mode: str
) -> AnnData:
    X = np.concatenate(predictions, axis=0)

    index = (
        data_module.adata.obs_names if infer_mode == "bag" else np.arange(X.shape[0])
    )
    obs = pd.DataFrame(index=index)

    var = pd.DataFrame(index=data_module.genes)

    uns = data_module.adata.uns
    if "spot_cell_map" in uns.keys():
        del uns["spot_cell_map"]

    obsm = (
        {"spatial": data_module.adata.obsm["spatial"]}
        if ("spatial" in data_module.adata.obsm_keys() and (infer_mode == "instance"))
        else {}
    )

    return AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        uns=uns,
    )


def add_information_cell_adata(pred_adata: AnnData) -> AnnData:
    cell_embedding_path = Path(pred_adata.uns["cell_embedding_path"])
    assert cell_embedding_path.exists(), f"File {cell_embedding_path} does not exist."

    # Load metadata
    h5_file = h5py.File(cell_embedding_path, mode="r", swmr=True)
    key_to_load = ["barcode", "label"]
    obs = pd.DataFrame(
        data={key: h5_file[key][:].squeeze() for key in key_to_load},
    )
    obs["class"] = obs["label"].map(REV_CLASS_LABELS)

    # Load spatial coordinates
    obsm = {"spatial": h5_file["coords"][:]}
    pred_adata.uns["patch_size_src"] = h5_file["embedding"].attrs["patch_size_src"]

    # Merge with predictions
    pred_adata.obs = obs
    pred_adata.obsm = obsm

    return pred_adata


def save_metrics(metrics: pd.DataFrame, config: DictConfig) -> None:
    output_dir = METRICS_DIR / config.save_dir_tag / config.data.dataset_handler
    print(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{config.exp_tag};test_slide={config.data.predict_id};infer_mode={config.infer_mode}.csv"
    )
    metrics.to_csv(output_path)


def save_adata_predictions(adata_pred: AnnData, config: DictConfig) -> None:
    output_dir = PREDS_DIR / config.save_dir_tag / config.data.dataset_handler
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{config.exp_tag};test_slide={config.data.predict_id};infer_mode={config.infer_mode}.h5ad"
    )
    adata_pred.uns["cell_embedding_path"] = str(adata_pred.uns["cell_embedding_path"])
    if config.infer_mode == "inference":
        adata_pred = add_information_cell_adata(adata_pred)
    adata_pred.write_h5ad(output_path)


def save_plots(
    metrics: pd.DataFrame, adata: AnnData, adata_pred: AnnData, config: DictConfig
) -> None:
    save_dir = (
        FIGURES_DIR
        / config.save_dir_tag
        / config.data.dataset_handler
        / f"{config.exp_tag};test_slide={config.data.predict_id};infer_mode={config.infer_mode}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = format_metric_df(metrics)
    pcc_metrics = metrics[metrics["metric"] == "scc"].sort_values(
        by="value", ascending=False
    )
    genes_to_plot = pcc_metrics["gene"].values[:10]
    for gene in genes_to_plot:
        if gene not in adata_pred.var_names:
            logger.info(f"Skipping {gene}, not found in adata pred")
            continue
        elif gene not in adata.var_names:
            logger.info(f"Skipping {gene}, not found in adata")
            continue
        else:
            logger.info(f"Plotting {gene}.")
            plot_top_genes(adata, adata_pred, gene, save_dir / f"{gene}.png")


def predict_and_save(
    config_dir: Path,
    config_kwargs: dict,
    infer_mode: str,
    compute_metrics: bool = False,
    save_adata: bool = False,
    with_plot: bool = False,
):
    assert infer_mode in [
        "bag",
        "instance",
        "inference",
    ], f"Invalid infer_mode: {infer_mode}"

    # Setup config
    config = load_yaml(config_dir / "config.yaml")
    config = update_config(config, config_kwargs)
    config.data.genes = config.model.gene_names
    config.infer_mode = infer_mode
    config.data.normalize = True
    config.data.log1p = True
    logger.info(f"Experiment tag for prediction: {config['exp_tag']}")

    # Load trained model
    model = load_model(config)
    if infer_mode in ["instance", "inference"] and isinstance(
        model, BaseLightningModel
    ):
        model.set_test_mode("instance")

    # Load data
    if compute_metrics & (infer_mode == "instance"):
        config.data.dataset_handler = "supervised"
    stage = "inference" if infer_mode == "inference" else "predict"
    data_module = prepare_data_module(
        config.data, stage=stage, task_type=config.model.task_type
    )

    # Predict
    trainer = Trainer()
    predictions = trainer.predict(model, dataloaders=data_module.predict_dataloader())
    adata_pred = format_predictions(predictions, data_module, infer_mode)
    logger.info(f"Predicted {adata_pred.shape} / {data_module.adata.shape} spots.")

    # Optionally save adata
    if save_adata:
        save_adata_predictions(adata_pred, config)

    # Optionally compute metrics
    if compute_metrics:
        metrics = compute_gene_metrics(data_module.adata, adata_pred)
        save_metrics(metrics, config)

        # Optionally save plots
        if with_plot:
            save_plots(metrics, data_module.adata, adata_pred, config)
