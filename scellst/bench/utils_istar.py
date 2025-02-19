import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("external/istar"))

import numpy as np
import pandas as pd
import requests
import shutil
import torch
import yaml
from anndata import AnnData
from numpy import ndarray
from PIL import Image
from lightning import seed_everything
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from scellst.constant import METRICS_DIR, MODELS_DIR
from scellst.dataset.data_module import prepare_data_module
from scellst.io_utils import load_yaml, load_config
from scellst.metrics.gene import compute_gene_metrics
from scellst.utils import update_config, create_tag

from external.istar.impute import get_data, SpotDataset, ForwardSumModel
from external.istar.utils import read_string

Image.MAX_IMAGE_PIXELS = 933120000
MODEL_NAME = "istar"


def read_yaml(filename: str) -> dict:
    with open(filename) as yaml_file:
        return yaml.safe_load(yaml_file)


# Function to download a file from a URL if it doesn't already exist
def download_file_if_not_exists(url: str, target_path: Path):
    if target_path.exists():
        logger.info(f"File {target_path} already exists. Skipping download.")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes
    with open(target_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    logger.info(f"File {target_path} downloaded successfully.")


def download_istar_checkpoints(checkpoint_path: Path):
    # Define the source URLs and target file paths
    source_256 = (
        "https://upenn.box.com/shared/static/p0hc12l1bpu5c7fzieotv1d6592btv1l.pth"
    )
    source_4k = (
        "https://upenn.box.com/shared/static/8qayhxzmdjpcr5loi88xtkfbqomag8a9.pth"
    )
    target_256 = checkpoint_path / "vit256_small_dino.pth"
    target_4k = checkpoint_path / "vit4k_xs_dino.pth"

    # Create the checkpoints directory if it doesn't exist
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Download the files if they don't already exist
    download_file_if_not_exists(source_256, target_256)
    download_file_if_not_exists(source_4k, target_4k)


def format_to_istar(config: DictConfig, working_dir: Path) -> None:
    assert (
        len(config.data.list_training_ids) == 1
    ), "Only implemented for 1 training slide with istar"

    # Prepare output folder
    config.istar_training_prefix = (
        f"{config.data.list_training_ids[0]}_{config.data.genes}"
    )
    config.istar_gene_names = config.data.genes
    print(config.istar_training_prefix)
    print(config.istar_gene_names)
    output_folder = working_dir / "data" / config.istar_training_prefix
    output_folder.mkdir(exist_ok=True)

    # Load data
    data_module = prepare_data_module(
        config.data, stage="fit", task_type=config.model.task_type
    )
    config.model.gene_names = data_module.get_gene_names()
    adata = data_module.adata_dict[config.data.list_training_ids[0]]

    # Copy gene names    # check that all genes in df are in adata.var_names
    with open((output_folder / "gene-names.txt"), "w") as f:
        for gene in adata.var_names.tolist():
            f.write(f"{gene}\n")

    # Prepare counts
    df_counts = pd.DataFrame(
        data=adata.X, index=adata.obs_names, columns=adata.var_names
    )
    df_counts = df_counts.reset_index()
    df_counts = df_counts.rename(columns={"index": "spot"})
    logger.info(f"Count matrix shape: {df_counts.shape}")
    df_counts.to_csv(output_folder / "cnts.tsv", sep="\t", index=False)

    # Copy image
    slide_name = (
        config.data.data_dir / "wsis" / f"{config.data.list_training_ids[0]}.tif"
    )
    assert slide_name.exists(), f"No slide found at {slide_name}"
    logger.info(f"Slide name: {slide_name}")
    img = Image.open(slide_name)
    img.save(output_folder / "he-raw.jpg", "JPEG", quality=100)

    # Prepare spot coordinates
    df_coords = pd.DataFrame(
        data=adata.obsm["spatial"], index=adata.obs_names, columns=["x", "y"]
    )
    df_coords = df_coords.reset_index()
    df_coords = df_coords.rename(columns={"index": "spot"})
    logger.info(f"Coordinate matrix shape: {df_coords.shape}")
    df_coords.to_csv(output_folder / "locs-raw.tsv", sep="\t", index=False)

    # Prepare pixel size
    key = next(iter(adata.uns["spatial"].keys()))
    with open(output_folder / "pixel-size-raw.txt", "w") as f:
        f.write(
            f'{8000 / 2000 * adata.uns["spatial"][key]["scalefactors"]["tissue_downscaled_fullres_scalef"]}'
        )

    # Prepare radius size
    with open(output_folder / "radius-raw.txt", "w") as f:
        f.write(
            f'{0.5 * adata.uns["spatial"][key]["scalefactors"]["spot_diameter_fullres"]}'
        )

    logger.info("Visium slide prepared to be used with istar.")


def launch_istar_scripts(
    prefix: str, working_dir: Path, device: str = "cuda", pixel_size: float = 0.5
):
    """
    Launches the given Bash script commands using the provided prefix, device, and pixel size.

    :param prefix: The prefix directory for the analysis (e.g., "data/demo/").
    :param device: The device to use for computation ("cuda" or "cpu").
    :param pixel_size: The desired pixel size for the whole analysis.
    :param working_dir: The working directory for the subprocess commands.
    """
    working_dir_path = working_dir
    prefix_path = Path("data") / prefix

    try:
        # Preprocess histology image
        pixel_size_file = working_dir / prefix_path / "pixel-size.txt"
        with pixel_size_file.open("w") as f:
            f.write(f"{pixel_size}\n")

        subprocess.run(
            ["python3", "-u", "rescale.py", str(prefix_path) + "/", "--image"],
            check=True,
            cwd=working_dir_path,
        )
        logger.info("Rescaling done.")
        subprocess.run(
            ["python3", "-u", "preprocess.py", str(prefix_path) + "/", "--image"],
            check=True,
            cwd=working_dir_path,
        )
        logger.info("Preprocessing done.")

        # Extract histology features
        subprocess.run(
            [
                "python3",
                "-u",
                "extract_features.py",
                str(prefix_path) + "/",
                f"--device={device}",
            ],
            check=True,
            cwd=working_dir_path,
        )
        logger.info("Feature extraction done.")

        # Auto detect tissue mask
        embeddings_hist_file = prefix_path / "embeddings-hist.pickle"
        mask_small_file = prefix_path / "mask-small.png"
        subprocess.run(
            [
                "python3",
                "-u",
                "get_mask.py",
                str(embeddings_hist_file),
                str(mask_small_file),
            ],
            check=True,
            cwd=working_dir_path,
        )
        logger.info("Masking done.")

        # Predict super-resolution gene expression
        subprocess.run(
            [
                "python3",
                "-u",
                "rescale.py",
                str(prefix_path) + "/",
                "--locs",
                "--radius",
            ],
            check=True,
            cwd=working_dir_path,
        )
        logger.info("Image super resolution done.")

        # Train gene expression prediction model and predict at super-resolution
        states_dir = working_dir_path / prefix_path / "states"
        print(states_dir)
        if states_dir.exists():
            logger.info("Removing trained istar.")
            shutil.rmtree(states_dir)
        subprocess.run(
            [
                "python3",
                "-u",
                "impute.py",
                str(prefix_path) + "/",
                "--epochs=400",
                "--n-states=1",
                f"--device={device}",
            ],
            check=True,
            cwd=working_dir_path,
        )
        logger.info("Training done.")

        logger.info("Script executed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


@torch.inference_mode()
def predict_spot(model: nn.Module, dataset: SpotDataset, device: str) -> ndarray:
    device = torch.device(device)
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    preds = []
    for x, _ in dataloader:
        super_pred = model(x.to(device))
        preds.append(super_pred.mean(-2))
    return torch.cat(preds).cpu().numpy()


def train_istar(
    config_path: Path, config_kwargs: dict, working_dir: Path = Path("external/istar")
) -> None:
    logger.info(f"Training: {MODEL_NAME}")

    # Load config to get data parameters
    config = load_config(config_path)
    config = update_config(config, config_kwargs)
    config.exp_tag = create_tag(config_kwargs)

    # Set seed
    seed_everything(config.data.seed)

    # Download checkpoints
    checkpoint_path = Path("external/istar/checkpoints")
    download_istar_checkpoints(checkpoint_path)

    # Format data
    format_to_istar(config, working_dir)

    # Launch istar scripts
    prefix = config.istar_training_prefix
    launch_istar_scripts(prefix, working_dir, device="cuda", pixel_size=0.5)

    # Save config
    output_dir = MODELS_DIR / MODEL_NAME / config.exp_tag
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_dir / "config.yaml")
    logger.info("Saved Model")


def eval_istar(
    config_dir: Path, config_kwargs: dict, working_dir: Path = Path("external/istar")
) -> None:
    logger.info(f"Evaluating: {MODEL_NAME}")

    # Setup config
    config = load_yaml(config_dir / "config.yaml")
    config = update_config(config, config_kwargs)
    config.data.genes = config.model.gene_names

    # Set seed
    seed_everything(config.data.seed)

    # Load data
    data_module = prepare_data_module(
        config.data, stage="predict", task_type=config.model.task_type
    )
    adata_truth = data_module.adata.copy()
    prefix = f"{config.data.predict_id}_{config.istar_gene_names}"

    # Load data
    embs, cnts, locs = get_data(str(working_dir / "data" / prefix) + "/")
    obs_names = cnts.index
    var_names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    factor = 16
    radius = int(read_string(working_dir / "data" / prefix / "radius.txt"))
    radius = radius / factor
    dataset = SpotDataset(embs, cnts, locs, radius)

    # Load model
    checkpoint_file = (
        working_dir / "data" / config.istar_training_prefix / "states/00/model.pt"
    )
    assert os.path.exists(checkpoint_file), "Trained model does not exist."
    logger.info(f"Loading trained model from {checkpoint_file}")
    model = ForwardSumModel.load_from_checkpoint(checkpoint_file)
    logger.info(f"Model loaded from {checkpoint_file}")
    model.eval()

    # Perform predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = predict_spot(model, dataset, device)

    # Format output
    adata_pred = AnnData(
        X=predictions,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )

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
