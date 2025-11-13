from matplotlib import pyplot as plt, cm
import scanpy as sc
from matplotlib.colors import Normalize
from scipy.stats import spearmanr, pearsonr
from anndata import AnnData
from pathlib import Path
from matplotlib.gridspec import GridSpec
from loguru import logger


def _rasterize_points_in_axes(ax: plt.Axes, rasterize: bool = True) -> None:
    """Rasterize only scatter-like PathCollections to keep text/axes vector."""
    if not rasterize:
        return
    for coll in getattr(ax, "collections", []):
        try:
            coll.set_rasterized(True)
        except Exception:
            pass

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

    _rasterize_points_in_axes(ax, rasterize=True)


def plot_top_genes(
    adata: AnnData, adata_pred: AnnData, list_gene: list[str], save_path: Path
) -> None:
    """
    Plot H&E image, spatial expression, and predicted vs true gene expression with a jointplot.
    """
    # Create a figure with gridspec for jointplot integration
    fig = plt.figure(figsize=(14, 5*len(list_gene)))
    gs = GridSpec(len(list_gene), 3, width_ratios=[1, 1, 0.1], figure=fig)

    for i, gene_name in enumerate(list_gene):
        logger.info(f"Plotting gene {gene_name}")

        # Extract data
        true_expression = adata[:, gene_name].X.flatten()
        predicted_expression = adata_pred[:, gene_name].X.flatten()

        # Compute Spearman correlation
        pcc = pearsonr(predicted_expression, true_expression)[0]

        # Subplot 2: Target gene spatial expression
        ax = fig.add_subplot(gs[i, 0])
        plot_spatial(adata, color=gene_name, title=f"Target gene {gene_name}", ax=ax)

        # Subplot 3: Predicted gene spatial expression
        ax = fig.add_subplot(gs[i, 1])
        plot_spatial(
            adata_pred,
            color=gene_name,
            title=f"Predicted gene {gene_name} (pcc: {pcc:.2f})",
            ax=ax,
        )

        # Colorbar
        ax = fig.add_subplot(gs[i, 2])
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.colormaps["magma"]
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=ax, orientation="vertical")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["low", "high"])

    # Save the figure
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
