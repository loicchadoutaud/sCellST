from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
from lightning.pytorch import seed_everything

from scellst.dataset.data_module import prepare_data_module
from scellst.trainer import prepare_trainer
from scellst.io_utils import load_config
from scellst.utils import update_config, create_tag, prepare_model


def train_and_save(
    config_path: str | Path,
    config_kwargs: dict,
) -> None:
    # Setup config
    config = load_config(config_path)
    config = update_config(config, config_kwargs)
    config.exp_tag = create_tag(config_kwargs)
    logger.info(f"Experiment tag: {config['exp_tag']}")

    # Seed everything
    seed_everything(config.data.seed)

    # Data loading and preprocessing
    data_module = prepare_data_module(
        config.data, stage="fit", task_type=config.model.task_type
    )

    # Build model
    config.model.gene_names = data_module.get_gene_names()
    config.predictor.output_dim = len(data_module.get_gene_names())
    model = prepare_model(config)

    # Final config
    logger.info(f"Config exp: {OmegaConf.to_yaml(config)}")

    # Trainer
    save_dir = Path("models") / "mil" / config.save_dir_tag / config.exp_tag
    trainer = prepare_trainer(config, save_dir)

    # Save config
    OmegaConf.save(config=config, f=save_dir / "config.yaml")

    # Train model
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )
    logger.success("Training complete.")
