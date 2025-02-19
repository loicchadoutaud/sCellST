import os

import h5py
import numpy as np
import pandas as pd
import scanpy as sc

from scellst.constant import REV_CLASS_LABELS, CLASS_LABELS, COLOR_MAP
from scellst.plots.plot_hest import (
    plot_segmentation_with_slide,
    save_class_cell_images,
    plot_spot_with_cells,
    save_cell_images,
    plot_visium,
)
from hest.HESTData import save_spatial_plot


class PlotMixin:
    def plot_cell_class_gallery(
        self,
        cell_img_save_dir: str,
        save_dir: str,
        name: str | None = None,
        n_cells: int = 100,
        suffix: str = "gallery",
    ) -> None:
        if name is None:
            name = self.meta["id"]

        h5_path = os.path.join(cell_img_save_dir, name + ".h5")
        assert os.path.exists(h5_path), f"{h5_path} does not exist"

        h5_file = h5py.File(h5_path, "r")
        cell_class = np.vectorize(REV_CLASS_LABELS.get)(h5_file["label"][:])
        all_cell_imgs = []
        unique_classes = np.sort(np.asarray(list(CLASS_LABELS.keys())))
        for class_name in unique_classes:
            # select random cell images from the class
            n_sample = min(n_cells, np.sum(cell_class == class_name))
            cell_idxs = np.random.choice(
                np.where(cell_class == class_name)[0], n_sample, replace=False
            )
            cell_idxs = np.sort(cell_idxs)
            all_cell_imgs.append(h5_file["img"][cell_idxs])
            if n_sample < n_cells:
                all_cell_imgs.append(
                    255
                    * np.ones(
                        (n_cells - n_sample, *h5_file["img"].shape[1:]),
                        dtype=h5_file["img"].dtype,
                    )
                )

        all_cell_imgs = np.concatenate(all_cell_imgs, axis=0)
        os.makedirs(save_dir, exist_ok=True)
        save_class_cell_images(
            all_cell_imgs,
            unique_classes,
            os.path.join(save_dir, name + f"_{suffix}.png"),
        )

    def plot_cell_random_gallery(
        self,
        cell_img_save_dir: str,
        save_dir: str,
        name: str | None = None,
        n_cells: int = 500,
        suffix: str = "gallery",
    ) -> None:
        if name is None:
            name = self.meta["id"]

        h5_path = os.path.join(cell_img_save_dir, name + ".h5")
        assert os.path.exists(h5_path), f"{h5_path} does not exist"

        h5_file = h5py.File(h5_path, "r")
        # select random cell images from the class
        n_sample = min(n_cells, len(h5_file["img"]))
        cell_idxs = np.random.choice(len(h5_file["img"]), n_sample, replace=False)
        cell_idxs = np.sort(cell_idxs)
        all_cell_imgs = h5_file["img"][cell_idxs]

        os.makedirs(save_dir, exist_ok=True)
        save_cell_images(
            all_cell_imgs,
            os.path.join(save_dir, name + f"_random_{suffix}.png"),
        )

    def plot_cell_gene_gallery(
        self,
        cell_img_save_dir: str,
        cell_adata_save_dir: str,
        save_dir: str,
        gene: str,
        name: str | None = None,
        n_cells: int = 100,
        suffix: str = "gallery",
    ) -> None:
        if name is None:
            name = self.meta["id"]

        h5_path = os.path.join(cell_img_save_dir, name + ".h5")
        assert os.path.exists(h5_path), f"{h5_path} does not exist"

        cell_adata_path = os.path.join(cell_adata_save_dir, name + ".h5ad")
        assert os.path.exists(cell_adata_path), f"{cell_adata_path} does not exist"

        cell_adata = sc.read_h5ad(cell_adata_path)
        df = cell_adata.to_df()[gene].copy()
        df.sort_values(inplace=True, ascending=False)
        cell_names = df.index[:n_cells]

        h5_file = h5py.File(h5_path, "r")
        cell_barcodes = h5_file["cell_barcode"][:].astype(str).flatten()
        cell_idxs = np.flatnonzero(np.isin(cell_barcodes, cell_names))
        all_cell_imgs = h5_file["img"][cell_idxs]

        os.makedirs(save_dir, exist_ok=True)
        save_cell_images(
            all_cell_imgs,
            os.path.join(save_dir, name + f"_{gene}_{suffix}.png"),
        )

    def plot_cell_visualisation(
        self,
        save_dir: str,
        name: str | None = None,
        shape_name: str = "cellvit",
        coordinates_name: str = "he",
        suffix: str = "spatial_vis",
    ) -> None:
        if name is None:
            name = self.meta["id"]

        gdf = self.get_shapes(shape_name, coordinates_name).shapes
        gdf["class_color"] = gdf["class"].map(COLOR_MAP)
        os.makedirs(save_dir, exist_ok=True)
        plot_segmentation_with_slide(
            img_arr=self.adata.uns["spatial"]["ST"]["images"]["downscaled_fullres"],
            gdf=gdf,
            save_path=os.path.join(save_dir, name + f"_{suffix}.png"),
        )

    def plot_spot_and_cell(
        self,
        cell_img_save_dir: str,
        save_dir: str,
        name: str | None = None,
        spot_idx: int = 0,
        shape_name: str = "cellvit",
        coordinates_name: str = "he",
    ) -> None:
        if name is None:
            name = self.meta["id"]

        h5_cell_path = os.path.join(cell_img_save_dir, name + ".h5")
        save_path = os.path.join(save_dir, name + f"_spot_{spot_idx}.png")

        # Get spot image
        patch_size_src = self.meta["spot_diameter"] / self.pixel_size
        coords_center = self.adata.obsm["spatial"][spot_idx]
        coords_topleft = coords_center - patch_size_src // 2
        coords_topleft = np.array(coords_topleft).astype(int)
        patch_size_src = int(patch_size_src)
        img_arr = self.wsi.read_region(
            location=coords_topleft, level=0, size=(patch_size_src, patch_size_src)
        )

        # Get cell segmentation on top of spot image
        xlim = (coords_topleft[0], coords_topleft[0] + patch_size_src)
        ylim = (coords_topleft[1], coords_topleft[1] + patch_size_src)
        gdf = self.get_shapes(shape_name, coordinates_name).shapes
        gdf["class_color"] = gdf["class"].map(COLOR_MAP)
        gdf["center"] = gdf.centroid
        gdf["center_x"] = gdf["center"].apply(lambda c: c.x)
        gdf["center_y"] = gdf["center"].apply(lambda c: c.y)
        spot_radius_px = 0.5 * self.meta["spot_diameter"] / self.pixel_size
        mask = (gdf["center_x"] - coords_center[0]) ** 2 + (
            gdf["center_y"] - coords_center[1]
        ) ** 2 <= spot_radius_px**2
        gdf_plot = gdf[mask].copy()
        gdf_plot["geometry"] = gdf_plot["geometry"].translate(
            xoff=-xlim[0], yoff=-ylim[0]
        )

        # Get cell belonging to spot
        h5_file = h5py.File(h5_cell_path, "r")
        cell_spots = h5_file["spot"][:].astype(str)
        cell_idxs = np.where(cell_spots == self.adata.obs_names[spot_idx])[0]
        cell_imgs = h5_file["img"][cell_idxs]

        assert len(cell_imgs) == len(
            gdf_plot
        ), f"Number of cells {len(cell_imgs)} does not match number of cells in spot {len(gdf_plot)}"
        os.makedirs(save_dir, exist_ok=True)
        plot_spot_with_cells(img_arr, gdf_plot, cell_imgs, save_path)

    def plot_spots_with_number_of_cells(
        self,
        cell_img_save_dir: str,
        save_dir: str,
        name: str | None = None,
    ) -> None:
        if name is None:
            name = self.meta["id"]

        h5_cell_path = os.path.join(cell_img_save_dir, name + ".h5")
        with h5py.File(h5_cell_path, "r") as h5_file:
            cell_spots = h5_file["spot"][:].astype(str).squeeze()

        spot_cell_counts = pd.Series(cell_spots).value_counts()
        spot_cell_counts.drop("None", inplace=True)
        self.adata.obs["spot_cell_count"] = spot_cell_counts

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, name + "_spot_cell_count.png")
        plot_visium(self.adata, save_path, "spot_cell_count")
