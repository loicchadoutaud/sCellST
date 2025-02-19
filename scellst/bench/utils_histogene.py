import argparse
import os
import shutil
import sys
from pathlib import Path

from lightning import seed_everything
from loguru import logger

sys.path.insert(0, os.path.abspath("external/HisToGene"))

import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from scellst.constant import MODELS_DIR, METRICS_DIR
from scellst.dataset.data_module import prepare_data_module
from scellst.io_utils import load_config, load_yaml
from scellst.utils import update_config, create_tag
from scellst.metrics.gene import compute_gene_metrics
from external.HisToGene.predict import model_predict
from external.HisToGene.vis_model import HisToGene

MODEL_NAME = "HisToGene"


def get_default_args():
    return argparse.Namespace(
        patch_size=112,
        n_genes=None,
        learning_rate=1.0e-5,
        max_epochs=100,
        n_pos=128,
        batch_size=1,
        n_layers=4,
    )


def train_histogene(config_path: Path, config_kwargs: dict) -> None:
    logger.info(f"Training: {MODEL_NAME}")

    # Load config to get data parameters
    config = load_config(config_path)
    config = update_config(config, config_kwargs)
    config.exp_tag = create_tag(config_kwargs)

    # Set seed
    seed_everything(config.data.seed)

    # Load method args
    args = get_default_args()

    # Load data
    data_module = prepare_data_module(
        config.data, stage="fit", task_type=config.model.task_type
    )
    config.model.gene_names = data_module.get_gene_names()

    # Prepare model
    args.n_genes = len(data_module.get_gene_names())
    model = HisToGene(
        patch_size=args.patch_size,
        n_genes=args.n_genes,
        learning_rate=args.learning_rate,
        n_pos=args.n_pos,
        n_layers=args.n_layers,
    )

    # Save model + config
    output_dir = MODELS_DIR / MODEL_NAME / config.exp_tag
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    trainer = pl.Trainer(devices=1, max_epochs=args.max_epochs, log_every_n_steps=1)
    trainer.fit(model, data_module.train_dataloader())

    # Save model
    trainer.save_checkpoint(output_dir / "best.pt")
    OmegaConf.save(config=config, f=output_dir / "config.yaml")
    print("Saved Model")


def eval_histogene(config_dir: Path, config_kwargs: dict) -> None:
    logger.info(f"Evaluating: {MODEL_NAME}")

    # Setup config
    config = load_yaml(config_dir / "config.yaml")
    config = update_config(config, config_kwargs)
    config.data.genes = config.model.gene_names

    # Set seed
    seed_everything(config.data.seed)

    # Load method args
    args = get_default_args()

    # Load data
    data_module = prepare_data_module(
        config.data, stage="predict", task_type=config.model.task_type
    )

    # Load trained model
    output_dir = MODELS_DIR / MODEL_NAME / config.exp_tag
    args.n_genes = len(data_module.get_gene_names())
    logger.info(f"Loading trained model from {output_dir}")
    model = HisToGene.load_from_checkpoint(
        output_dir / "best.pt",
        patch_size=args.patch_size,
        n_genes=args.n_genes,
        learning_rate=args.learning_rate,
        n_pos=args.n_pos,
    )
    device = torch.device("cpu")
    adata_pred, adata_truth = model_predict(
        model, data_module.predict_dataloader(), attention=False, device=device
    )
    adata_pred.var_names = data_module.get_gene_names()
    adata_truth.var_names = data_module.get_gene_names()
    logger.info(f"Predicted {adata_pred.shape} / {adata_truth.shape} spots.")

    # Compute metrics
    metrics = compute_gene_metrics(adata_truth, adata_pred)
    output_dir = METRICS_DIR / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{config.exp_tag};test_slide={config.data.predict_id};model={MODEL_NAME}.csv"
    )
    metrics.to_csv(output_path)
