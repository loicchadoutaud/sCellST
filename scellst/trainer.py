import os
import shutil
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig


def clean_output_folder(output_folder: Path) -> None:
    if os.path.exists(output_folder):
        logger.info(f"Found existing output folder: {output_folder}, cleaning it...")
        shutil.rmtree(output_folder, ignore_errors=True)


def prepare_callbacks(output_folder: str, patience: int):
    es_callback = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_folder,
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    return [es_callback, checkpoint_callback, lr_callback]


def prepare_trainer(config: DictConfig, save_dir: Path) -> L.Trainer:
    clean_output_folder(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    logger = WandbLogger(project="CellST", save_dir=save_dir, name=save_dir.name)
    callbacks = prepare_callbacks(save_dir, config.trainer.patience)

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epoch,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        devices=1,
        log_every_n_steps=10,
    )
    return trainer
