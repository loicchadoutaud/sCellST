import random
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
# import histomicstk as htk
# from histomicstk.preprocessing.color_deconvolution import stain_unmixing_routine
# from histomicstk.preprocessing.color_normalization import (
#     deconvolution_based_normalization,
# )
from loguru import logger
from numpy import ndarray
from torchvision.transforms.v2 import Transform

from scellst.constant import DATA_DIR


def compute_target_representative_stains(
    h5_path: Path,
    key: str = "img",
    num_samples: int = 100,
    output_path: Path = Path("target_representative_img.png"),
):
    """
    Computes and saves a single target representative image
    from a random sample of images in an H5 file.
    This image will be used by HistomicsTK to fit the normalizer.

    Args:
        h5_path (Path): Path to the H5 file containing source images.
        key (str): The key for the image dataset within the H5 file.
        num_samples (int): Number of images to sample for creating the average target image.
                           HistomicsTK's Macenko fit expects a single image.
        output_image (Path): Path to save the resulting representative stain.
    """
    logger.info(f"Computing target representative image from {h5_path}...")
    stain_unmixing_routine_params = {
        "stains": ["hematoxylin", "eosin"],
        "stain_unmixing_method": "macenko_pca",
    }
    with h5py.File(h5_path, "r") as f:
        dataset = f[key]
        if num_samples > len(dataset):
            print(
                f"Warning: num_samples ({num_samples}) > dataset size ({len(dataset)}). Using all images."
            )
            num_samples = len(dataset)

        # Randomly select indices
        indices = random.sample(range(len(dataset)), num_samples)

        # Load selected images and convert to numpy array
        selected_images = [dataset[i] for i in indices]
        selected_images = np.array(selected_images)  # Shape (num_samples, H, W, C)
        list_stain_matrix = [
            stain_unmixing_routine(target_img, **stain_unmixing_routine_params)
            for target_img in selected_images
        ]

        # Compute the mean image
        average_stain_matrix = np.mean(list_stain_matrix, axis=0)

    logger.info(f"Saving average stain to {output_path}...")
    output_folder = DATA_DIR / "rep_stains"
    output_folder.mkdir(parents=True, exist_ok=True)
    np.save(output_folder / output_path, average_stain_matrix)


def batch_normalize_from_h5(
    h5_path: Path,
    target_path: Path,
    save_path: Path,
    num_samples: int = 5,
    key: str = "img",
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Applies Macenko normalization to a batch of images from an H5 file using HistomicsTK.

    Returns:
        A tuple containing two lists: original images and normalized images.
    """
    random.seed(seed)

    # Load the target representative image for fitting the normalizer
    logger.info(
        f"Loading target stain from {target_path} for normalizer fitting..."
    )
    norm = MacenkoStainNormalization(target_stain_path=target_path)

    # Initialize MacenkoNormalizer
    # HistomicsTK's MacenkoNormalizer computes the target stain matrix and concentrations
    # from the provided target image during initialization or a fit method.

    with h5py.File(h5_path, "r") as f:
        dataset = f[key]
        indices = random.sample(range(len(dataset)), num_samples)

        originals = []
        normalized = []
        for i in indices:
            src_img = dataset[i].astype(np.uint8)

            # Transform the image
            nrm_img = norm(src_img)
            originals.append(src_img)
            normalized.append(nrm_img)

    plot_comparison(originals, normalized, save_path)


def plot_comparison(originals: list[np.ndarray], normalized: list[np.ndarray], save_path: Path):
    """Plots original and normalized images side-by-side."""
    n_rows = 10
    n_cols = 10
    n = n_rows // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 20))

    for i in range(len(originals)):
        axs[i // n, 2 * (i % n)].imshow(originals[i])
        axs[i // n, 2 * (i % n)].set_title(f"Original {i + 1}")
        axs[i // n, 2 * (i % n)].axis("off")

        axs[i // n, 2 * (i % n) + 1].imshow(normalized[i])
        axs[i // n, 2 * (i % n) + 1].set_title(f"Normalized {i + 1}")
        axs[i // n, 2 * (i % n) + 1].axis("off")

    fig.savefig(save_path, bbox_inches="tight", dpi=100)


class MacenkoStainNormalization(Transform):
    def __init__(self, target_stain_path: Path):
        super().__init__()
        logger.info(f"Loading representative stain from {target_stain_path}...")
        self.W_target = np.load(target_stain_path)

    def __call__(self, img: ndarray) -> ndarray:
        return deconvolution_based_normalization(img, W_target=self.W_target)


if __name__ == "__main__":
    data_path = Path("../../hest_data/cell_images")
    output_path = Path("target_representative_stain.npy")

    target_id = "TENX39"
    src_id = "TENX65"

    target_h5_file = data_path / "TENX39_cellvit.h5"
    source_h5_file = data_path / "TENX65_hoverfast.h5"

    compute_target_representative_stains(
        h5_path=target_h5_file,
        key="img",
        num_samples=100,  # Use more samples for a robust average target image
        output_path=output_path,
    )

    batch_normalize_from_h5(
        h5_path=source_h5_file,
        target_path=DATA_DIR / "rep_stains" / output_path,
        save_path=DATA_DIR / "rep_images" / f"{src_id}_{target_id}.png",
        num_samples=50,
    )


