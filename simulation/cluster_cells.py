import argparse
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms
from PIL import Image
from anndata import AnnData
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import Tensor
from torchvision import transforms
from torchvision.utils import make_grid

def load_single_file(file: str) -> ndarray:
    return np.load(file)


def load_single_image(file: str) -> Tensor:
    trans = [torchvision.transforms.CenterCrop(48), transforms.ToTensor()]
    image = transforms.Compose(trans)(Image.open(file))
    return image


def load_dataset_h5(output_dir: str) -> ndarray:
    print("Loading embeddings h5...")
    h5file = h5py.File(output_dir, "r")
    data = np.array(h5file["embeddings"])
    print("Done...")
    return data


def load_slide_outputs(
    slide_name: str,
    exp_folder_path: str,
    n_cells: int,
) -> tuple[AnnData, list[str]]:
    # Find annotation file
    annotation_names = [
        f
        for f in os.listdir(os.path.join(exp_folder_path, "csv_annotation"))
        if (f.startswith(slide_name) & f.endswith(".csv"))
    ]
    assert len(annotation_names) > 0, "No slide name found"
    annotation_path = os.path.join(
        exp_folder_path, "csv_annotation", annotation_names[0]
    )
    assert os.path.exists(annotation_path), "No annotation files found"
    annotations = pd.read_csv(annotation_path, index_col=0)

    # Subset annotations if too large
    if len(annotations) > n_cells:
        idx_cells = np.random.choice(len(annotations), size=n_cells, replace=False)
    else:
        idx_cells = np.arange(len(annotations))
    annotations = annotations.iloc[idx_cells]

    # Load embeddings and store them within an AnnData
    embedding_folder = os.path.join(exp_folder_path, "cell_embeddings", slide_name)
    embedding_paths = [
        os.path.join(embedding_folder, f)
        for f in os.listdir(embedding_folder)
        if f.endswith(".h5")
    ]
    list_tags = [
        os.path.splitext(f)[0]
        for f in os.listdir(embedding_folder)
        if f.endswith(".h5")
    ]
    print(list_tags)

    # Create AnnData
    adata = AnnData(obs=annotations)
    adata.obs["class"] = adata.obs["class"].astype("category")
    adata.obs["class"] = adata.obs["class"].cat.reorder_categories(
        np.sort(adata.obs["class"].unique())
    )
    adata.obs["slide"] = slide_name

    # Add embeddings
    for path in embedding_paths:
        print(f"Start loading tag: {os.path.basename(path)}")
        tag = os.path.splitext(os.path.basename(path))[0]
        adata.obsm[f"X_{tag}"] = load_dataset_h5(path)[idx_cells]
        adata.obs["img_path"] = os.path.join(
            exp_folder_path, "cell_images", slide_name + ".h5"
        )
        adata.obs["cell_index"] = idx_cells
        print("End loading")

    print(f"Loaded data for slide {slide_name}: {adata.shape}")
    return adata, list_tags


def load_all_slide_outputs(
    list_slide_name: list[str],
    exp_folder_path: str,
    n_cells: int = 1_000_000,
) -> AnnData:
    # Load all slide outputs
    list_adata = []
    print(list_slide_name)
    for slide_name in list_slide_name:
        print(f"Start loading slide: {slide_name}")
        adata, list_tags = load_slide_outputs(
            slide_name, exp_folder_path, n_cells // len(list_slide_name)
        )
        list_adata.append(adata)

    # Create final AnnData
    adata = ad.concat(list_adata)
    adata.obs_names_make_unique()
    adata.uns["list_tags"] = list_tags
    print(f"Loaded adata with all slides: {adata.shape}")
    return adata


def load_images(adata: AnnData) -> Tensor:
    list_image_dataset_path = (
        adata.obs["img_path"].drop_duplicates().sort_values().tolist()
    )
    list_image_name = [
        os.path.splitext(os.path.basename(f))[0] for f in list_image_dataset_path
    ]
    dict_dataset = {
        slide_name: h5py.File(dataset_path, "r")
        for slide_name, dataset_path in zip(list_image_name, list_image_dataset_path)
    }
    images = []
    for i in range(len(adata)):
        row = adata.obs.iloc[i]
        images.append(dict_dataset[row["slide"]]["images"][int(row["cell_index"])])
    data = torch.from_numpy(np.stack(images))
    data = torch.swapaxes(data, 1, 3)
    return data


def compute_embeddings(adata: AnnData, obsm_key: str) -> None:
    print(f"Processing {obsm_key}")

    reducer = PCA(n_components=50)
    adata.obsm[f"X_pca_{obsm_key}"] = reducer.fit_transform(adata.obsm[f"X_{obsm_key}"])
    print("pca embeddings computed.")


def compute_all_embeddings(adata: AnnData) -> None:
    for tag in adata.uns["list_tags"]:
        compute_embeddings(adata, tag)


def cluster_embeddings(adata: AnnData, obsm_key: str, n_clusters: int = 20) -> None:
    # create a kmeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    adata.obs[f"kmeans_{obsm_key}"] = kmeans.fit_predict(
        adata.obsm[f"X_pca_{obsm_key}"]
    )
    adata.uns[f"kmeans_{obsm_key}"] = kmeans
    print("Kmeans fitted.")

    # find the closest cells to each cluster
    adata.obsm[f"kmeans_{obsm_key}_dist"] = kmeans.transform(
        adata.obsm[f"X_pca_{obsm_key}"]
    )


def cluster_all_embeddings(adata: AnnData) -> None:
    for tag in adata.uns["list_tags"]:
        cluster_embeddings(adata, tag)


def prepare_image_grid(
    adata: AnnData, obsm_key: str, cluster_key: str, cluster: int, n_cells: int = 100
) -> Tensor:
    # Find cells to plot
    idx_closest = np.argsort(adata.obsm[f"{cluster_key}_{obsm_key}_dist"], axis=0)[
        :n_cells, cluster
    ].flatten()

    # Load images
    images = load_images(adata[adata.obs_names[idx_closest]])

    return make_grid(images, nrow=10)


def plot_cell_gallery(
    adata: AnnData, save_dir: str, obsm_key: str, cluster_key: str = "kmeans"
) -> None:
    n_rows, n_cols = (len(adata.obs[f"{cluster_key}_{obsm_key}"].unique()) * 2) // 8, 8
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12 * (n_cols // 2), 6 * n_rows),
        gridspec_kw={"width_ratios": [3, 1] * (n_cols // 2)},
        layout="constrained",
    )
    axs = axs.flatten()

    all_classes = adata.obs["class"].drop_duplicates().sort_values().tolist()
    all_slides = adata.obs["slide"].drop_duplicates().sort_values().tolist()
    print(f"Clustering results: {adata.obs[f'{cluster_key}_{obsm_key}'].value_counts()}")
    for label in adata.obs[f"{cluster_key}_{obsm_key}"].unique():
        image_grid = prepare_image_grid(adata, obsm_key, cluster_key, label)
        axs[label * 2].imshow(image_grid.permute(1, 2, 0))
        axs[label * 2].axis("off")
        axs[label * 2].set_title(f"Cluster {label}", fontsize=20)

        # add histogram of cell types and slide names
        df_plot = adata[adata.obs[f"{cluster_key}_{obsm_key}"] == label].obs
        sns.countplot(
            data=df_plot,
            y="class",
            order=all_classes,
            hue="slide",
            hue_order=all_slides,
            ax=axs[(label * 2) + 1],
        )
        axs[(label * 2) + 1].tick_params(labelsize=20)
        axs[(label * 2) + 1].set(xlabel=None, ylabel=None)
        axs[(label * 2) + 1].get_legend().remove()

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.savefig(os.path.join(save_dir, f"cell_gallery_{cluster_key}_{obsm_key}.png"), dpi=300, bbox_inches="tight")
    print("Cell gallery plotted.")


def plot_all_visualisations(
    adata: AnnData,
    output_folder_path: str,
    cluster_key: str = "kmeans",
) -> None:
    os.makedirs(output_folder_path, exist_ok=True)

    for tag in adata.uns["list_tags"]:
        plot_cell_gallery(adata, output_folder_path, tag, cluster_key)


def save_embeddings_for_simuation(
    list_slide_name: list[str],
    embedding_folder_path: str,
    output_folder_path: str,
    n_cells: int = 1_000_000,
) -> None:
    os.makedirs(output_folder_path, exist_ok=True)

    # Set seeds
    np.random.seed(0)

    # Load data into an anndata
    adata = load_all_slide_outputs(
        list_slide_name, embedding_folder_path, n_cells,
    )

    # Compute low dimension embeddings
    compute_all_embeddings(adata)

    # Compute clustering
    cluster_all_embeddings(adata)

    # Create plots
    plot_all_visualisations(adata, output_folder_path)

    os.makedirs(os.path.join(output_folder_path, "cell_adata"), exist_ok=True)
    uns = adata.uns
    adata.uns = {}
    print(f"Saving anndata to {output_folder_path}")
    adata.write_h5ad(os.path.join(output_folder_path, "cell_adata", f"{os.path.splitext(list_slide_name[0])[0]}.h5ad"))
    adata.uns = uns
    print("anndata saved.")


if __name__ == "__main__":
    print("Starting main script...")
    parser = argparse.ArgumentParser(description="Analysing SSL embeddings")
    parser.add_argument(
        "--list_slide_name", nargs="+", help="Name of slides to analyse."
    )
    parser.add_argument("--embedding_folder_path", type=str, help="Path to outputs")
    parser.add_argument(
        "--output_folder_path", help="number of desired clusters"
    )
    args = parser.parse_args()
    save_embeddings_for_simuation(
        list_slide_name=args.list_slide_name,
        embedding_folder_path=args.embedding_folder_path,
        output_folder_path=args.output_folder_path,
    )
    print("End of python script.")
