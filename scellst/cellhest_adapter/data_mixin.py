import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from tqdm.auto import tqdm

from scellst.cellhest_adapter.cell_utils import (
    find_spot_containing_cells_vectorized,
    predict_cell_dataset,
    compute_mean_std,
)
from scellst.constant import CLASS_LABELS, REV_CLASS_LABELS, DATA_DIR


class DataMixin:
    def dump_cell_images(
        self,
        save_dir: str,
        name: str | None = None,
        target_patch_size: int = 72,
        target_pixel_size: float = 0.25,
        shape_name: str = "cellvit",
        coordinates_name: str = "he",
        write_in_tmp_dir: bool = False,
    ):
        """Dump H&E patches centered around cells to a .h5 file.

            Patches are computed such that:
             - each cell is rescaled to `target_pixel_size` um/px
             - a crop of `target_patch_size`x`target_patch_size` pixels around each segmented cell is derived (from cellVIT segmentation).

        Args:
            save_dir (str): directory where the .h5 cell file will be saved
            name (str, optional): file will be saved as {name}.h5. Defaults to 'cell'.
            target_patch_size (int, optional): target cell size in pixels (after scaling to match `target_pixel_size`). Defaults to 48.
            target_pixel_size (float, optional): target patch pixel size in um/px. Defaults to 0.25.
            shape_name (str, optional): name of the shape. Defaults to 'cellvit'.
            coordinates_name (str, optional): name of the coordinates. Defaults to 'he'.
            verbose (int, optional): verbosity level. Defaults to 0.
        """
        if name is None:
            name = self.meta["id"]

        dst_pixel_size = target_pixel_size

        gdf = self.get_shapes(shape_name, coordinates_name).shapes

        src_pixel_size = self.pixel_size

        patch_size_src = target_patch_size * (dst_pixel_size / src_pixel_size)
        coords_center = np.stack([gdf.centroid.x, gdf.centroid.y], axis=1)
        spot_radius_px = 0.5 * self.meta["spot_diameter"] / src_pixel_size
        logger.info(f"spot_radius_px: {spot_radius_px}")
        spot_containing_cells = find_spot_containing_cells_vectorized(
            self.adata, coords_center, spot_radius_px
        )
        coords_topleft = coords_center - patch_size_src // 2
        len_tmp = len(coords_topleft)
        in_slide_mask = (
            (0 <= coords_topleft[:, 0] + patch_size_src)
            & (coords_topleft[:, 0] < self.wsi.width)
            & (0 <= coords_topleft[:, 1] + patch_size_src)
            & (coords_topleft[:, 1] < self.wsi.height)
        )
        coords_topleft = coords_topleft[in_slide_mask]
        if len(coords_topleft) < len_tmp:
            warnings.warn(
                f"Filtered {len_tmp - len(coords_topleft)} cells outside the WSI"
            )

        coords_topleft = np.array(coords_topleft).astype(int)
        patcher = self.wsi.create_patcher(
            target_patch_size,
            src_pixel_size,
            dst_pixel_size,
            custom_coords=coords_topleft,
            threshold=0,
        )

        extra_assets = {
            "barcode": gdf.index.values[in_slide_mask],
            "spot": spot_containing_cells[in_slide_mask],
        }
        if "cell_id" in gdf.columns:
            extra_assets["id"] = gdf["cell_id"].values[in_slide_mask]
        else:
            extra_assets["id"] = gdf.index.values[in_slide_mask]
        if "class" in gdf.columns:
            extra_assets["label"] = gdf["class"].map(CLASS_LABELS).values[in_slide_mask]
        else:
            extra_assets["label"] = np.full(len(gdf), -1)[in_slide_mask]

        os.makedirs(save_dir, exist_ok=True)
        h5_path = os.path.join(save_dir, name + ".h5")

        if write_in_tmp_dir:
            with tempfile.TemporaryDirectory() as tmp_dir:
                logger.info(f"Using temp dir: {tmp_dir}")
                h5_tmp_path = os.path.join(tmp_dir, name + ".h5")
                patcher.to_h5(
                    h5_tmp_path,
                    extra_assets=extra_assets,
                )
                shutil.copy(h5_tmp_path, save_dir)
        else:
            patcher.to_h5(
                h5_path,
                extra_assets=extra_assets,
            )

    def dump_cell_image_stats(
        self,
        cell_img_save_dir: str,
        save_dir: str,
        name: str | None = None,
    ):
        if name is None:
            name = self.meta["id"]

        h5_path = os.path.join(cell_img_save_dir, name + ".h5")
        norm_dict = compute_mean_std(h5_path)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, name + ".json")
        with open(save_path, "w") as f:
            json.dump(norm_dict, f)

    def dump_cell_embeddings(
        self,
        cell_img_save_dir: str,
        cell_stat_img_save_dir: str,
        normalisation_type: str,
        save_dir: str,
        name: str | None = None,
        model_name: str = "resnet50",
        weights_path: str = "imagenet",
        tag: str = "",
        write_in_tmp_dir: bool = False,
    ) -> None:
        if name is None:
            name = self.meta["id"]

        os.makedirs(save_dir, exist_ok=True)
        dataset_path = os.path.join(cell_img_save_dir, name + ".h5")
        match normalisation_type:
            case "self":
                stats_path = os.path.join(cell_stat_img_save_dir, name + ".json")
            case "train":
                if "moco" in weights_path:
                    stats_path = (
                        Path(weights_path).parent / "moco_model_best_mean_std.json"
                    )
                else:
                    stats_path = DATA_DIR / "imagenet_stats.json"
            case _:
                raise ValueError(f"{normalisation_type} should be either train or self")
        logger.info(f"Using {stats_path} for image normalisation.")

        assert os.path.exists(dataset_path), f"{dataset_path} does not exist"
        embedding_path = os.path.join(save_dir, f"{tag}_{name}.h5")
        logger.info(f"Saving embeddings to {embedding_path}")

        if write_in_tmp_dir:
            with tempfile.TemporaryDirectory() as tmp_dir:
                logger.info(f"Using temp dir: {tmp_dir}")
                embedding_tmp_path = os.path.join(tmp_dir, f"{tag}_{name}.h5")
                predict_cell_dataset(
                    dataset_path,
                    stats_path,
                    model_name,
                    weights_path,
                    embedding_tmp_path,
                    "cuda" if torch.cuda.is_available() else "cpu",
                )
                shutil.copy(embedding_tmp_path, embedding_path)
        else:
            predict_cell_dataset(
                dataset_path,
                stats_path,
                model_name,
                weights_path,
                embedding_path,
                "cuda" if torch.cuda.is_available() else "cpu",
            )

    def get_embeddings(
        self,
        cell_emb_dir: str,
        tag: str,
        n_cell_max: int = 100_000,
        batch_loading_size: int = 10_000,
    ) -> AnnData:
        name = self.meta["id"]

        embedding_path = os.path.join(cell_emb_dir, f"{tag}_{name}.h5")
        assert os.path.exists(embedding_path), f"{embedding_path} does not exist"

        with h5py.File(embedding_path, mode="r") as h5file:
            n_cell = len(h5file["embedding"])
            if n_cell > n_cell_max:
                idxs = np.random.choice(a=n_cell, size=n_cell_max, replace=False)
                idxs = np.sort(idxs)
            else:
                idxs = np.arange(n_cell)

            # Batch-wise loading
            logger.info("Start loading...")
            X, barcodes, labels = [], [], []
            for start in tqdm(range(0, len(idxs), batch_loading_size)):
                end = start + batch_loading_size
                X.append(h5file["embedding"][idxs[start:end]])
                barcodes.append(h5file["barcode"][idxs[start:end]].squeeze())
                labels.append(h5file["label"][idxs[start:end]].squeeze())
            X = np.vstack(X)
            barcodes = np.concatenate(barcodes)
            labels = np.concatenate(labels)
            logger.info("Loading done.")

        classes = pd.Series(labels).map(REV_CLASS_LABELS)
        obs = pd.DataFrame(
            np.stack([labels, classes, barcodes, idxs], axis=1),
            columns=["label", "class", "barcode", "id"],
        )

        return AnnData(X, obs=obs)
