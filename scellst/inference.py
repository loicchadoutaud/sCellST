import logging
import os.path

import lightning as L
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from lightning import Trainer
from lightning.pytorch import seed_everything
from pandas import DataFrame
from torch.utils.data import DataLoader

from scellst import REGISTRY_KEYS
from scellst.config import Config
from scellst.dataset.data_utils import select_labels, load_anndata, preprocess_adata
from scellst.model.CellST import CellST
from scellst.model.Supervised import Supervised
from scellst.train import prepare_dataset_processer, prepare_anndata
from scellst.utils import read_yaml
from scripts.eval_visium_predictions import eval_predictions

logger = logging.getLogger(__name__)


def load_pretrained_mil_model(exp_path: str) -> CellST:
    # Find model path
    model_path = os.path.join(exp_path, "best_model.ckpt")
    assert os.path.join(model_path), "Trained model not found"

    # Load model
    return CellST.load_from_checkpoint(model_path)


def load_pretrained_supervised_model(exp_path: str) -> Supervised:
    # Find model path
    model_path = os.path.join(exp_path, "best_model.ckpt")
    assert os.path.join(model_path), "Trained model not found"

    # Load model
    return Supervised.load_from_checkpoint(model_path)


def compute_spot_predictions(dataloader: DataLoader, model: CellST, trainer: Trainer) -> DataFrame:
    predictions = trainer.predict(model, dataloader)
    predictions = torch.cat([pred[0][REGISTRY_KEYS.OUTPUT_PREDICTION] for pred in predictions]).numpy()
    return pd.DataFrame(data=predictions, index=dataloader.dataset.spot_names, columns=model.gene_names)


def compute_cell_predictions(dataloader: DataLoader, model: CellST, trainer: Trainer) -> AnnData:
    predictions = trainer.predict(model, dataloader)
    gene_predictions = torch.cat([pred[REGISTRY_KEYS.OUTPUT_PREDICTION] for pred in predictions]).numpy()
    adata = AnnData(gene_predictions)
    adata.var_names = model.gene_names
    adata.obs["label"] = dataloader.dataset.instance_labels
    adata.obs["class"] = dataloader.dataset.instance_class
    return adata


def compute_cell_supervised_predictions(dataloader: DataLoader, model: Supervised, trainer: Trainer) -> AnnData:
    predictions = trainer.predict(model, dataloader)
    gene_predictions = torch.cat([pred[REGISTRY_KEYS.OUTPUT_PREDICTION] for pred in predictions]).numpy()
    adata = AnnData(gene_predictions)
    adata.var_names = model.gene_names
    return adata


def infer_spot(exp_path: str, eval_folder_path: str, output_path: str) -> None:
    output_path = os.path.join(output_path, "spot")
    os.makedirs(output_path, exist_ok=True)

    # Load conf
    conf_dict = read_yaml(os.path.join(exp_path, "parameters.yaml"))
    config = Config(**conf_dict)

    # Seed everything
    seed_everything(config.seed, workers=True)

    # Load anndata
    adata = prepare_anndata(eval_folder_path, config.preprocessing_config, "mil/adata_with_mil.h5ad")
    adata = select_labels(adata, config.preprocessing_config)

    # Load model
    model = load_pretrained_mil_model(exp_path)
    trainer = L.Trainer(devices=1)

    # Prepare despot_dataset processor
    dataset_processor = prepare_dataset_processer(config.data_config)
    config.predictor_config.output_dim = adata.shape[1]

    # Prepare dataloaders
    spot_dataloader = dataset_processor.prepare_spot_inference_dataloader(adata, eval_folder_path, os.path.join(exp_path, "normalisation.yaml"))

    # Perform spot inference
    predictions = compute_spot_predictions(spot_dataloader, model, trainer)
    common_genes = list(set(model.gene_names).intersection(adata.var_names))
    common_genes.sort()
    logger.info(f"Found {len(common_genes)} / {len(model.gene_names)} in anndata.")
    adata = adata[:, common_genes]
    adata.layers["predictions"] = predictions[common_genes]

    # Save anndata
    adata.uns["exp_id"] = config.trainer_config.exp_id
    adata.uns["slide_name"] = os.path.basename(eval_folder_path)
    output_filename = f"slide-{os.path.basename(eval_folder_path)};{config.trainer_config.exp_id};model-sCellST.h5ad"
    adata.write_h5ad(os.path.join(output_path, output_filename))

    logger.info("Inference script ended without errors.")


def infer_cell(exp_path: str, eval_folder_path: str, output_path: str) -> None:
    output_path = os.path.join(output_path, "cell")
    os.makedirs(output_path, exist_ok=True)

    # Load conf
    conf_dict = read_yaml(os.path.join(exp_path, "parameters.yaml"))
    config = Config(**conf_dict)

    # Seed everything
    seed_everything(config.seed, workers=True)

    # Load anndata
    adata = prepare_anndata(eval_folder_path, config.preprocessing_config, "mil/adata_with_mil.h5ad")
    adata = select_labels(adata, config.preprocessing_config)

    # Load model
    model = load_pretrained_mil_model(exp_path)
    trainer = L.Trainer(devices=1)

    # Prepare despot_dataset processor
    dataset_processor = prepare_dataset_processer(config.data_config)

    # Prepare dataloaders
    cell_dataloader = dataset_processor.prepare_cell_inference_dataloader(adata.uns["MIL"]["cell_label"], os.path.join(eval_folder_path, "mil"))

    # Perform cell inference
    cell_adata = compute_cell_predictions(cell_dataloader, model, trainer)
    common_genes = list(set(model.gene_names).intersection(adata.var_names))
    common_genes.sort()
    logger.info(f"Found {len(common_genes)} / {len(model.gene_names)} in anndata.")
    cell_adata = cell_adata[:, common_genes]
    cell_adata.obsm["spatial"] = adata.uns["MIL"]["cell_coordinates"]

    # Save anndata
    cell_adata.uns["exp_id"] = config.trainer_config.exp_id
    cell_adata.uns["slide_name"] = os.path.basename(eval_folder_path)
    output_filename = f"slide-{os.path.basename(eval_folder_path)};{config.trainer_config.exp_id};model-sCellST.h5ad"
    cell_adata.write_h5ad(os.path.join(output_path, output_filename))

    logger.info("Inference script ended without errors.")


def infer_cell_supervised_exp(exp_path: str, eval_folder_path: str, output_path: str) -> None:
    output_path = os.path.join(output_path, "cell")
    os.makedirs(output_path, exist_ok=True)

    # Load conf
    conf_dict = read_yaml(os.path.join(exp_path, "parameters.yaml"))
    config = Config(**conf_dict)

    # Seed everything
    seed_everything(config.seed, workers=True)

    # Load anndata
    adata = load_anndata(os.path.join(eval_folder_path, "cell_adata_labels.h5ad"))
    adata = preprocess_adata(
        adata, config.preprocessing_config.normalize, config.preprocessing_config.log1p, config.preprocessing_config.filtering
    )
    adata = select_labels(adata, config.preprocessing_config)

    # Load model
    model = load_pretrained_supervised_model(exp_path)
    trainer = L.Trainer(devices=1)

    # Prepare despot_dataset processor
    dataset_processor = prepare_dataset_processer(config.data_config)

    # Prepare dataloaders
    cell_dataloader = dataset_processor.prepare_cell_supervised_inference_dataloader(adata)

    # Perform cell inference
    cell_adata = compute_cell_supervised_predictions(cell_dataloader, model, trainer)
    common_genes = list(set(model.gene_names).intersection(adata.var_names))
    common_genes.sort()
    logger.info(f"Found {len(common_genes)} / {len(model.gene_names)} in anndata.")
    cell_adata = cell_adata[:, common_genes]

    # Save anndata
    cell_adata.uns["exp_id"] = config.trainer_config.exp_id
    cell_adata.uns["slide_name"] = os.path.basename(eval_folder_path)
    output_filename = f"slide-{os.path.basename(eval_folder_path)};{config.trainer_config.exp_id};model-SupervisedST.h5ad"
    cell_adata.write_h5ad(os.path.join(output_path, output_filename))

    logger.info("Inference script ended without errors.")