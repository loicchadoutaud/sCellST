from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from loguru import logger
from matplotlib import pyplot as plt
from numpy import ndarray
from torchvision.utils import make_grid

from scellst.constant import CELL_IMG_DIR, CLASS_LABELS


def prepare_image_grid(
    adata: AnnData, path_dataset: Path, cluster: int, n_cells: int, cluster_key: str
) -> ndarray:
    # Find cells to plot
    idx_closest = np.argsort(adata.obsm[f"{cluster_key}_dist"], axis=0)[
        :n_cells, cluster
    ].flatten()
    cell_ids = adata[idx_closest].obs["id"].values
    cell_slides = adata[idx_closest].obs["slide"].values

    # Load images
    images = []
    for id, slide in zip(cell_ids, cell_slides):
        image_dataset_path = path_dataset / CELL_IMG_DIR / f"{slide}.h5"
        assert (
            image_dataset_path.exists()
        ), f"Image dataset not found: {image_dataset_path}"
        with h5py.File(image_dataset_path, "r") as h5file:
            images.append(torch.from_numpy(h5file["img"][int(id)]))
    images = torch.stack(images).permute(0, 3, 1, 2)
    return (make_grid(images, nrow=10)).permute(1, 2, 0).numpy()


def plot_ssl_cell_gallery(
    adata: AnnData,
    path_dataset: Path,
    tag: str,
    save_path: Path,
    n_cells_per_cluster: int = 100,
    cluster_key: str = "kmeans_detailed",
) -> None:
    n_rows, n_cols = (len(adata.obs[cluster_key].unique()) * 2) // 8 + 1, 8
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12 * (n_cols // 2), 6 * n_rows),
        gridspec_kw={"width_ratios": [3, 1] * (n_cols // 2)},
        layout="constrained",
    )
    axs = axs.flatten()

    all_labels = adata.obs[cluster_key].drop_duplicates().sort_values().tolist()
    all_classes = CLASS_LABELS.keys()
    all_slides = adata.obs["slide"].drop_duplicates().sort_values().tolist()
    for i, label in enumerate(all_labels):
        image_grid = prepare_image_grid(
            adata, path_dataset, label, n_cells_per_cluster, cluster_key
        )
        axs[i * 2].imshow(image_grid)
        axs[i * 2].axis("off")
        axs[i * 2].set_title(f"Cluster {label}", fontsize=20)

        # add histogram of cell types and slide names
        df_plot = adata[adata.obs[cluster_key] == label].obs
        sns.countplot(
            df_plot,
            y="class",
            order=all_classes,
            hue="slide",
            hue_order=all_slides,
            ax=axs[(i * 2) + 1],
        )
        axs[(i * 2) + 1].tick_params(labelsize=20)
        axs[(i * 2) + 1].set(xlabel=None, ylabel=None)
        axs[(i * 2) + 1].get_legend().remove()

    handles, labels = axs[(i * 2) + 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")

    for j in range((i + 1) * 2, len(axs)):
        axs[j].axis("off")

    fig.savefig(save_path / f"cell_gallery_kmeans_{tag}.png")
    logger.info("Cell gallery plotted.")


def plot_umap(adata: AnnData, tag: str, save_path: Path) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(24, 4))
    for i, obs_key in enumerate(["kmeans", "slide", "technology", "class"]):
        sc.pl.umap(adata, color=obs_key, ax=axs[i], show=False)
        axs[i].set_title(obs_key)
    plt.tight_layout()
    fig.savefig(save_path / f"umap_{tag}.png")
    logger.info("Umap plotted.")
