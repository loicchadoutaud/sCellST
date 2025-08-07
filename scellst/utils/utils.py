import os
import socket
import subprocess
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from omegaconf import DictConfig
from pandas import Series

from scellst.constant import REV_CLASS_LABELS
from scellst.lightning_model.gene_lightning_model import GeneLightningModel
from scellst.lightning_model.base_lightning_model import BaseLightningModel
from simulation.supervised_lightning_model import SupervisedLightningModel


def split_to_dict(s: Series):
    return dict(item.split("=") for item in s.split(";"))


def is_port_available(port: int):
    """Check if a port is available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def find_available_port(start: int = 1024, end: int = 65535):
    """Find an available port in the given range."""
    ports = np.arange(start, end)
    np.random.shuffle(ports)
    for port in ports:
        if is_port_available(port):
            print(f"Using port : {port}")
            return port
    raise RuntimeError("No available port found in the range.")


def run_moco_script(
    tag: str,
    list_slides: list[str],
    path_dataset: str,
    n_gpus: int,
    n_cpus_per_gpu: int,
):
    # Define the base path to the script
    script_path = os.path.abspath(os.path.join("external", "cell_SSL", "main_moco.py"))
    model_output_path = os.path.abspath(os.path.join("models", "ssl"))
    os.makedirs(model_output_path, exist_ok=True)

    default_port = find_available_port()

    # Construct the command as a list of arguments
    command = [
        "python3",
        script_path,
        "-b",
        "4096",
        "--epochs",
        "150",
        "--workers",
        str(n_gpus * n_cpus_per_gpu),
        "--dist-url",
        f"tcp://localhost:{default_port}",
        "--world-size",
        "1",
        "--multiprocessing-distributed",
        "--rank",
        "0",
        "--tag",
        tag,
        "--n_cell_max",
        "1_000_000",
        "--list_slides",
        *list_slides,
        "--data_path",
        path_dataset,
        "--output_path",
        model_output_path,
    ]

    # Run the command
    try:
        _ = subprocess.run(command, check=True, text=True)
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")


def create_tag(config: dict) -> str:
    if "data_dir" in config.keys():
        del config["data_dir"]
    if "save_dir_tag" in config.keys():
        del config["save_dir_tag"]
    if "list_training_ids" in config.keys():
        if not "fold" in config.keys():
            config["train_slide"] = next(iter(config["list_training_ids"]))
        del config["list_training_ids"]
    return ";".join([f"{key}={value}" for key, value in config.items()])


def update_config(config: DictConfig, updates: dict) -> DictConfig:
    """
    Update a nested configuration dictionary (or DictConfig) with the provided updates.

    Args:
        config (DictConfig): The original configuration object.
        updates (dict): A dictionary of updates, where keys are simple keys
                        that may correspond to any level in the config.

    Returns:
        DictConfig: The updated configuration object.

    Raises:
        KeyError: If a key in updates is not found in the config.
    """

    def find_and_update(d, target_key, value):
        """Recursively searches for the key in the nested structure and updates its value."""
        if target_key in d:
            d[target_key] = value
            return True
        for key, sub_d in d.items_ex(resolve=False):  # Avoid resolving unfilled values
            if isinstance(sub_d, (dict, DictConfig)):
                if find_and_update(sub_d, target_key, value):
                    return True
        return False

    for key, value in updates.items():
        if not find_and_update(config, key, value):
            raise KeyError(f"Key '{key}' not found in the configuration.")

    return config


def load_model(
    config: DictConfig, save_path: Path = Path("models") / "mil"
) -> BaseLightningModel:
    checkpoint_path = (
        save_path / config.save_dir_tag / config.exp_tag / "best_model.ckpt"
    )
    if config.data.dataset_handler == "supervised":
        model = SupervisedLightningModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
    else:
        model = GeneLightningModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    if config.data.dataset_handler == "xenium":
        model.set_test_mode("instance")
    return model


def prepare_model(config: DictConfig) -> L.LightningModule:
    # Adjust input dim
    if "rn18" in config.data.embedding_tag:
        config.predictor.input_dim = 512
    elif "one-hot-celltype" in config.data.embedding_tag:
        config.predictor.input_dim = 6
    else:
        config.predictor.input_dim = 2048
    logger.info(f"Using input dimension: {config.predictor.input_dim}")

    # Set model parameters
    if config.data.dataset_handler == "supervised":
        logger.info(f"Using supervised lightning module")
        model = SupervisedLightningModel(
            predictor_config=config.predictor, **config.model
        )
        return model

    # Create model
    return GeneLightningModel(predictor_config=config.predictor, **config.model)


def add_information_cell_adata(pred_adata: AnnData) -> AnnData:
    cell_embedding_path = Path(pred_adata.uns["cell_embedding_path"])
    assert cell_embedding_path.exists(), f"File {cell_embedding_path} does not exist."

    # Load metadata
    h5_file = h5py.File(cell_embedding_path, mode="r")
    key_to_load = ["barcode", "label"]
    obs = pd.DataFrame(
        data={key: h5_file[key][:].squeeze() for key in key_to_load},
    )
    obs["class"] = obs["label"].map(REV_CLASS_LABELS)

    obsm = {"spatial": h5_file["spatial"][:]}

    # Merge with predictions
    pred_adata.obs = obs
    pred_adata.obsm = obsm

    return pred_adata
