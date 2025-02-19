from matplotlib import pyplot as plt, cm
import scanpy as sc
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
from anndata import AnnData
from pathlib import Path
from matplotlib.gridspec import GridSpec


def plot_spatial(
    adata: AnnData,
    color: str | None,
    title: str,
    ax: plt.Axes,
    img_key: str = "downscaled_fullres",
) -> None:
    """
    Helper function to plot spatial data using scanpy.
    """
    # Remove color from background
    if color is None:
        img_key = img_key
    else:
        img_key = None
        ax.set_facecolor("black")

    # Check if visium or xenium
    if len(adata) > 6000:
        size = 0.3
        spot_size = None
    else:
        size = 1.0
        spot_size = 80

    sc.pl.spatial(
        adata,
        color=color,
        title=title,
        img_key=img_key,
        size=size,
        spot_size=spot_size,
        show=False,
        color_map="magma",
        vmin="p1",
        vmax="p99",
        ax=ax,
        colorbar_loc=None,
    )
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_top_genes(
    adata: AnnData, adata_pred: AnnData, gene_name: str, save_path: Path
) -> None:
    """
    Plot H&E image, spatial expression, and predicted vs true gene expression with a jointplot.
    """
    # Extract data
    true_expression = adata[:, gene_name].X.flatten()
    predicted_expression = adata_pred[:, gene_name].X.flatten()

    # Compute Spearman correlation
    scc = spearmanr(predicted_expression, true_expression)[0]

    # Prepare figure size
    img_shape = adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"].shape

    # Create a figure with gridspec for jointplot integration
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.1], figure=fig)

    # Subplot 2: Target gene spatial expression
    ax = fig.add_subplot(gs[0])
    plot_spatial(adata, color=gene_name, title=f"Target gene {gene_name}", ax=ax)

    # Subplot 3: Predicted gene spatial expression
    ax = fig.add_subplot(gs[1])
    plot_spatial(
        adata_pred,
        color=gene_name,
        title=f"Predicted gene {gene_name} (scc: {scc:.2f})",
        ax=ax,
    )

    # Colorbar
    ax = fig.add_subplot(gs[2])
    norm = Normalize(vmin=0, vmax=1)
    cmap = plt.colormaps["magma"]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=ax, orientation="vertical")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])

    # Save the figure
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=100)
