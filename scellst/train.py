import logging
import os
import shutil
from dataclasses import asdict

import lightning as L
from anndata import AnnData
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from scellst.config import Config, PreprocessingConfig, DataConfig, TrainerConfig
from scellst.dataset.base_anndata_processer import BaseAnndataProcessor
from scellst.dataset.data_utils import (
    load_anndatas,
    preprocess_anndatas,
    prepare_mil_anndatas,
    load_anndata,
    prepare_mil_adata,
    preprocess_adata,
    select_labels,
    select_labels_multiple_slides,
)
from scellst.dataset.reg_anndata_processer import RegDatasetProcesser
from scellst.model.Supervised import Supervised
from scellst.module.instance_mil_model import InstanceMilModel
from scellst.module.loss import loss_dict
from scellst.model.CellST import CellST
from scellst.module.instance_distribution_mil import InstanceDistributionMilModel
from scellst.module.predictor import DistributionPredictor, Predictor
from scellst.module.supervised_model import SupervisedModel
from scellst.type import TaskType


logger = logging.getLogger(__name__)


def prepare_anndata(
    data_folder: str, config_preprocessing: PreprocessingConfig, path_adata: str
) -> AnnData:
    adata = load_anndata(os.path.join(data_folder, path_adata))
    logger.info("AnnData loaded.")
    adata = prepare_mil_adata(adata, config_preprocessing)
    logger.info("AnnData prepared.")
    return preprocess_adata(
        adata, config_preprocessing.normalize, config_preprocessing.log1p, config_preprocessing.filtering
    )


def prepare_anndatas(
    list_data_folder: list[str], config_preprocessing: PreprocessingConfig
) -> dict[str, AnnData]:
    dict_adata = load_anndatas(list_data_folder)
    dict_adata = prepare_mil_anndatas(dict_adata, config_preprocessing)
    return preprocess_anndatas(dict_adata, config_preprocessing)


def prepare_dataset_processer(data_config: DataConfig) -> BaseAnndataProcessor:
    if data_config.task_type in [
        TaskType.regression,
        TaskType.nb_total_regression,
        TaskType.nb_mean_regression,
    ]:
        logger.info("Using regression despot_dataset processor")
        DatasetProcesserClass = RegDatasetProcesser
    else:
        raise ValueError(f"{data_config.task_type} must be in {TaskType.list()}")
    return DatasetProcesserClass(**asdict(data_config))


def prepare_mil_model(config: Config) -> CellST:
    # Prepare predictor
    if config.task_type in [
        TaskType.nb_total_regression,
        TaskType.nb_mean_regression,
    ]:
        PredictorClass = DistributionPredictor
        ModelClass = InstanceDistributionMilModel
    else:
        PredictorClass = Predictor
        ModelClass = InstanceMilModel
    predictor = PredictorClass(**asdict(config.predictor_config))

    # Prepare criterion
    criterion = loss_dict[config.loss]
    mil_model = ModelClass(predictor, criterion, **asdict(config.model_config))

    return CellST(
        mil_model,
        lr=config.trainer_config.lr,
        gene_names=config.preprocessing_config.gene_to_pred,
    )


def prepare_supervised_model(config: Config) -> Supervised:
    if config.task_type in [
        TaskType.nb_total_regression,
        TaskType.nb_mean_regression,
    ]:
        PredictorClass = DistributionPredictor
    else:
        PredictorClass = Predictor
    predictor = PredictorClass(**asdict(config.predictor_config))

    ModelClass = SupervisedModel
    criterion = loss_dict[config.loss]
    supervised_model = ModelClass(predictor, criterion)

    return Supervised(
        supervised_model,
        lr=config.trainer_config.lr,
        gene_names=config.preprocessing_config.gene_to_pred,
    )


def prepare_trainer(config_trainer: TrainerConfig, save_path: str) -> L.Trainer:
    output_folder = os.path.join(save_path, config_trainer.exp_id)

    # Clean output folder
    if os.path.exists(output_folder):
        shutil.rmtree(os.path.join(output_folder, "lightning_logs"), ignore_errors=True)
        if os.path.exists(os.path.join(output_folder, "best_model.ckpt")):
            os.remove(os.path.join(output_folder, "best_model.ckpt"))

    # Prepare callbacks
    logger = TensorBoardLogger(
        save_dir=output_folder,
    )
    es_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=config_trainer.patience
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_folder,
        filename="best_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks = [es_callback, checkpoint_callback]

    # Prepare trainer
    trainer = L.Trainer(
        max_epochs=config_trainer.max_epoch,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        devices=1,
        log_every_n_steps=10,
    )

    return trainer


def train(
    config: Config,
    train_folder_path: str | list[str],
    save_path: str = "lightning",
    path_adata: str = "mil/adata_with_mil.h5ad",
) -> None:
    logger.info(f"Starting experiment: {config.trainer_config.exp_id}...")

    # Seed everything
    seed_everything(config.seed, workers=True)

    # Load anndata
    if isinstance(train_folder_path, str):
        adata = prepare_anndata(
            train_folder_path, config.preprocessing_config, path_adata
        )
        adata = select_labels(adata, config.preprocessing_config)
    else:
        raise ValueError(f"Got {type(train_folder_path)} for train_folder_path.")

    # Prepare despot_dataset processor
    dataset_processor = prepare_dataset_processer(config.data_config)
    dataset_processor.prepare_folds(adata)
    config.predictor_config.output_dim = adata.shape[1]

    # Prepare dataloaders
    train_loader, val_loader = dataset_processor.prepare_mil_training_dataloaders(
        adata, train_folder_path
    )

    # Prepare model
    model = prepare_mil_model(config)

    # Prepare trainer
    trainer = prepare_trainer(config.trainer_config, save_path)

    # Fit
    trainer.fit(model, train_loader, val_loader)


def train_supervised(
    config: Config,
    train_folder_path: str,
    save_path: str = "lightning",
    path_adata: str = "mil/adata_with_mil.h5ad",
) -> None:
    logger.info(f"Starting experiment: {config.trainer_config.exp_id}...")

    # Seed everything
    seed_everything(config.seed, workers=True)

    # Load anndata
    adata = load_anndata(os.path.join(train_folder_path, path_adata))
    adata = preprocess_adata(
        adata, config.preprocessing_config.normalize, config.preprocessing_config.log1p, config.preprocessing_config.filtering
    )
    adata = select_labels(adata, config.preprocessing_config)

    # Prepare despot_dataset processor
    dataset_processor = prepare_dataset_processer(config.data_config)
    dataset_processor.prepare_folds(adata)
    config.predictor_config.output_dim = adata.shape[1]

    # Prepare dataloaders
    train_loader, val_loader = dataset_processor.prepare_supervised_training_dataloaders(
        adata, train_folder_path
    )

    # Prepare model
    model = prepare_supervised_model(config)

    # Prepare trainer
    trainer = prepare_trainer(config.trainer_config, save_path)

    # Fit
    trainer.fit(model, train_loader, val_loader)
