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
from scellst.utils.io_utils import load_yaml
from scellst.lightning_model.base_lightning_model import BaseLightningModel
from scellst.metrics.gene import compute_gene_metrics
from scellst.metrics.metric_utils import format_metric_df
from scellst.plots.plot_spatial import plot_top_genes
from scellst.utils.utils import update_config, load_model


def format_predictions(predictions: list[Tensor], data_module: STDataModule) -> AnnData:
    X = np.concatenate(predictions, axis=0)

    index = data_module.get_obs_names()
    obs = pd.DataFrame(index=index)
    var = pd.DataFrame(index=data_module.genes)

    uns = data_module.adata.uns
    if "spot_cell_map" in uns.keys():
        del uns["spot_cell_map"]

    return AnnData(
        X=X,
        obs=obs,
        var=var,
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
    obs[["x", "y"]] = h5_file["coords"][:]
    obs["barcode"] = obs["barcode"].astype(str) + f"_{pred_adata.uns['hest_id']}"
    obs = obs.set_index("barcode")
    obs["class"] = obs["label"].map(REV_CLASS_LABELS)
    obs["class"] = obs["class"].fillna("Nolabel")
    pred_adata.obs = pred_adata.obs.join(obs, how="left")

    # Cell seg information
    pred_adata.uns["patch_size_src"] = h5_file["embedding"].attrs["patch_size_src"]

    # Spatial coords
    pred_adata.obsm["spatial"] = pred_adata.obs[["x", "y"]].values
    pred_adata.obs.drop(["x", "y"], axis=1, inplace=True)

    return pred_adata


def save_metrics(metrics: pd.DataFrame, config: DictConfig) -> None:
    output_dir = METRICS_DIR / config.save_dir_tag / config.data.dataset_handler
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
    metrics = format_metric_df(metrics, metric_list=["scc"])
    metrics.sort_values(by="scc", ascending=False, inplace=True)
    genes_to_plot = metrics["gene"].values[:10]
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
    align: bool = True,
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
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy="single_device",
    )
    predictions = trainer.predict(model, dataloaders=data_module.predict_dataloader())
    adata = data_module.adata
    adata_pred = format_predictions(predictions, data_module)
    logger.info(f"Predicted {adata_pred.shape}")

    # Find common observations
    if align:
        common_obs = list(set(adata.obs_names) & set(adata_pred.obs_names))
        logger.info(
            f"Found {len(common_obs)} / {len(adata.obs_names)} in measured cells."
        )
        logger.info(
            f"Found {len(common_obs)} / {len(adata_pred.obs_names)} in predicted cells."
        )
        adata = adata[common_obs, :]
        adata_pred = adata_pred[common_obs, :]
        adata_pred.obs = adata.obs
        adata_pred.obsm["spatial"] = adata.obsm["spatial"]
        adata_pred.uns = adata.uns

    # Optionally save adata
    if save_adata:
        save_adata_predictions(adata_pred, config)

    # Optionally compute metrics
    if compute_metrics:
        metrics = compute_gene_metrics(adata, adata_pred)
        save_metrics(metrics, config)

        # Optionally save plots
        if with_plot:
            metrics["tag"] = config["exp_tag"]
            save_plots(metrics, adata, adata_pred, config)
