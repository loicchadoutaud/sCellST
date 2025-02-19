import numpy as np
from geopandas import GeoDataFrame
import scanpy as sc
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from numpy import ndarray


def save_cell_images(img_arr: ndarray, save_path: str) -> None:
    """Save tensor cell images to a single image file.
    Images are saved first in rows and then in columns.

    Args:
        img_arr (ndarray): arr of cell images
        classes (ndarray): arr of cell classes
        save_path (str): path to save the image
    """
    n_rows = 10
    n_cols = len(img_arr) // n_rows + 1
    fig = plt.figure(figsize=(n_cols, n_rows))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1, direction="column"
    )
    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def save_class_cell_images(img_arr: ndarray, classes: ndarray, save_path: str) -> None:
    """Save tensor cell images to a single image file.
    Images are saved first in rows and then in columns.

    Args:
        img_arr (ndarray): arr of cell images
        classes (ndarray): arr of cell classes
        save_path (str): path to save the image
    """
    n_rows = 10
    n_cols = 50
    fig = plt.figure(figsize=(n_cols, n_rows))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1, direction="column"
    )
    for ax, im in zip(grid, img_arr):
        ax.imshow(im)
        ax.axis("off")

    # Add top lines
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    top_axes_columns = [axes[0] for axes in grid.axes_column]
    coordinates = list(map(get_bbox, top_axes_columns))
    coordinates = np.stack([np.asarray(coords).flatten() for coords in coordinates])

    y = 0.91
    y_text = y + 0.04
    x_threshold = 2 * (coordinates[1, 0] - coordinates[0, 2])
    for i in range(5):
        x_min, x_max = coordinates[10 * i, 0], coordinates[10 * (i + 1) - 1, 2]
        x_min, x_max = x_min + x_threshold, x_max - x_threshold
        line = plt.Line2D(
            [x_min, x_max],
            [y, y],
            transform=fig.transFigure,
            color="black",
            linewidth=15,
        )
        fig.add_artist(line)

        fig.text(
            x=np.mean([x_min, x_max]),
            y=y_text,
            s=classes[i],
            fontsize=50,
            transform=fig.transFigure,
            ha="center",
            va="center",
        )

    fig.savefig(save_path)
    plt.close(fig)


def plot_segmentation_with_slide(
    img_arr: ndarray, gdf: GeoDataFrame, save_path: str
) -> None:
    """Plot the segmentation on top of the slide image.

    Args:
        img_arr (ndarray): slide image
        gdf (GeoDataFrame): geodataframe with the segmentation
        save_path (str): path to save the image
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(img_arr)
    axs[0].axis("off")
    gdf.centroid.plot(ax=axs[1], color=gdf["class_color"], markersize=0.003)
    axs[1].invert_yaxis()
    axs[1].axis("off")
    fig.savefig(save_path)
    plt.close(fig)


def plot_spot_with_cells(
    spot_img: ndarray,
    gdf: GeoDataFrame,
    cell_imgs: ndarray,
    save_path: str,
) -> None:
    """Plot the spot image with the cell images.

    Args:
        spot_img (ndarray): spot image
        gdf (GeoDataFrame): geodataframe with the cell segmentation
        cell_imgs (ndarray): cell images
        save_path (str): path to save the image
    """
    fig = plt.figure(layout="constrained", figsize=(12, 6))
    subfigs = fig.subfigures(1, 2)
    ax1 = subfigs[0].subplots()
    ax1.imshow(spot_img)
    gdf.boundary.plot(ax=ax1, color=gdf["class_color"], linewidth=1)
    ax1.axis("off")
    ax1.set_title("Spot image with cell segmentation", fontsize=16)

    n_cells = cell_imgs.shape[0]
    n_rows = 6
    n_cols = n_cells // n_rows + 1
    grid = ImageGrid(
        subfigs[1], 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.05, share_all=True
    )
    for ax, im in zip(grid, cell_imgs):
        ax.imshow(im)
        ax.axis("off")
    for ax in grid[len(cell_imgs) :]:
        ax.axis("off")
    grid.axes_all[0].set_title(
        f"All cells (n={len(gdf)}) in spot", fontsize=16, loc="left"
    )
    fig.savefig(save_path)
    plt.close(fig)


def plot_visium(adata: sc.AnnData, save_path: str, key="total_counts") -> None:
    fig = sc.pl.spatial(
        adata,
        show=False,
        img_key="downscaled_fullres",
        color=[key],
        title=f"{key}",
        return_fig=True,
    )
    fig.savefig(save_path)
    plt.close(fig)
