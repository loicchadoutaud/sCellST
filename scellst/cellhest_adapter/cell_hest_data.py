import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe import from_pandas
from loguru import logger
import scanpy as sc

from scellst.cellhest_adapter.aggregation import count_transcripts
from scellst.cellhest_adapter.data_mixin import DataMixin
from scellst.cellhest_adapter.h5_utils import subset_h5files
from scellst.cellhest_adapter.plot_mixin import PlotMixin
from hest import (
    HESTData,
    XeniumHESTData,
    find_first_file_endswith,
    register_downscale_img,
)


class CellHESTData(DataMixin, PlotMixin, HESTData):
    @classmethod
    def from_HESTData(cls, st: HESTData):
        return CellHESTData(
            adata=st.adata,
            img=st.wsi,
            pixel_size=st.pixel_size,
            meta=st.meta,
            tissue_contours=st._tissue_contours,
            shapes=st.shapes,
        )


class CellXeniumHESTData(DataMixin, PlotMixin, XeniumHESTData):
    @classmethod
    def from_XeniumHESTData(cls, hest_dir: str, st: HESTData):
        id = st.meta["id"]
        transcripts_path = find_first_file_endswith(
            os.path.join(hest_dir, "transcripts"), f"{id}_transcripts.parquet"
        )
        t1 = time.time()
        transcript_df = pd.read_parquet(
            transcripts_path, columns=["cell_id", "feature_name", "qv", "he_x", "he_y"]
        )
        t2 = time.time()
        logger.info(f"Reading transcripts took {t2 - t1:.2f} seconds")
        return CellXeniumHESTData(
            adata=st.adata,
            img=st.wsi,
            pixel_size=st.pixel_size,
            meta=st.meta,
            tissue_contours=st._tissue_contours,
            shapes=st.shapes,
            transcript_df=transcript_df,
        )

    def _get_true_pixel_size(self) -> float:
        return self.meta.get("pixel_size", self.pixel_size)

    def _create_count_matrix(
        self,
        shape_name: str,
        value_key: str = "feature_name",
        coordinates_name: str = "he",
    ) -> AnnData:
        # Get cell transcripts
        df = self.transcript_df[self.transcript_df["cell_id"] != "UNASSIGNED"].copy()
        df = df[
            ~df["feature_name"]
            .astype(str)
            .str.startswith(("NegControlProbe", "antisense", "BLANK"))
        ]

        # Filter low quality transcripts
        n_transcripts = len(df)
        df = df[df["qv"] > 20]
        n_final_transcripts = len(df)
        logger.info(
            f"Kept {n_final_transcripts / n_transcripts * 100: .3f}% high quality transcripts"
        )

        # Get cell / nucleus geodataframe
        logger.info(f"Using {shape_name} shape")
        gdf = self.get_shapes(shape_name, coordinates_name).shapes

        # Count transcripts
        dd = from_pandas(df)

        # Cell expression matrix
        adata = count_transcripts(gdf, dd, value_key)

        logger.info(f"Initial adata object {adata}")
        return adata

    def _add_cell_seg_info(
        self,
        adata: AnnData,
        shape_name: str,
        coordinates_name: str = "he",
    ) -> AnnData:
        gdf = self.get_shapes(shape_name, coordinates_name).shapes
        gdf.index = gdf.index.astype(str)
        common_cell_idx = list(set(adata.obs_names).intersection(gdf.index))
        common_cell_idx.sort()
        logger.info(f"Found {len(common_cell_idx)} / {len(gdf.index)} in nuc gdf.")
        logger.info(f"Found {len(common_cell_idx)} / {len(adata)} in adata.")
        adata = adata[common_cell_idx].copy()
        gdf = gdf.loc[common_cell_idx]

        # Add spatial coordinates
        coords_center = np.stack([gdf.centroid.x, gdf.centroid.y], axis=1)
        adata.obsm["spatial"] = coords_center

        return adata

    def dump_cell_exp_matrix(
        self,
        save_dir: Path,
        shape_name: str,
        name: str | None = None,
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

        # Get cell expression
        cell_adata = self._create_count_matrix(shape_name)

        # Add cell seg stats
        cell_adata = self._add_cell_seg_info(cell_adata, shape_name)

        # Add downscaled image
        register_downscale_img(cell_adata, self.wsi, self._get_true_pixel_size())

        # Store pixel_size
        cell_adata.uns["pixel_size"] = self._get_true_pixel_size()

        # Inspect the AnnData object
        logger.info(f"Final adata object {cell_adata}")

        # Save the AnnData object
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}_{shape_name}.h5ad"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp_file:
            tmp_path = tmp_file.name
        try:
            cell_adata.write_h5ad(tmp_path)
            shutil.move(tmp_path, save_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def dump_cell_images_dataset(
        self,
        save_dir: Path,
        adata_dir: Path,
        shape_name: str,
        target_patch_size: int = 72,
        target_pixel_size: float = 0.25,
        name: str | None = None,
        write_in_tmp_dir: bool = False,
    ):
        """Dump H&E patches centered around cells to a .h5 file.

            Patches are computed such that:
             - each cell is rescaled to `target_pixel_size` um/px
             - a crop of `target_patch_size`x`target_patch_size` pixels around each segmented cell is derived (from cellVIT segmentation).

        Args:
            save_dir (Path): directory where the .h5 cell file will be saved
            name (str, optional): file will be saved as {name}.h5. Defaults to 'cell'.
            target_patch_size (int, optional): target cell size in pixels (after scaling to match `target_pixel_size`). Defaults to 48.
            target_pixel_size (float, optional): target patch pixel size in um/px. Defaults to 0.25.
            shape_name (str, optional): name of the shape. Defaults to 'cellvit'.
            coordinates_name (str, optional): name of the coordinates. Defaults to 'he'.
            verbose (int, optional): verbosity level. Defaults to 0.
        """
        logger.info("Saving cell images...")

        logger.info(f"Using destination pixel size: {target_pixel_size} um/px for patch size: {target_patch_size}.")
        if name is None:
            name = self.meta["id"]

        # Get cell geodataframe
        adata = sc.read_h5ad(adata_dir / f"{name}_{shape_name}.h5ad", backed="r")

        # Prepare image coordinates
        src_pixel_size = self.meta["pixel_size_um_estimated"]
        patch_size_src = target_patch_size * (target_pixel_size / src_pixel_size)
        logger.info(f"Found {src_pixel_size} µm/pixel, patch size {patch_size_src}.")
        coords_center = adata.obsm["spatial"].copy()
        coords_topleft = coords_center - patch_size_src // 2
        coords_topleft = np.round(coords_topleft).astype(int)

        # Filter cells outside of slide
        in_slide_mask = (
            (0 <= coords_topleft[:, 0] + patch_size_src)
            & (coords_topleft[:, 0] < self.wsi.width)
            & (0 <= coords_topleft[:, 1] + patch_size_src)
            & (coords_topleft[:, 1] < self.wsi.height)
        )
        if in_slide_mask.sum() < len(in_slide_mask):
            logger.info(
                f"Some cells {len(in_slide_mask) - in_slide_mask.sum()} are outside the slide, rewrite filtered adata"
            )
        adata = adata[in_slide_mask]
        cell_barcodes = adata.obs_names.tolist()
        coords_topleft = coords_topleft[in_slide_mask]
        coords_topleft = np.array(coords_topleft).astype(int)

        # Create patcher
        patcher = self.wsi.create_patcher(
            target_patch_size,
            src_pixel_size,
            target_pixel_size,
            custom_coords=coords_topleft,
        )

        extra_assets = {"barcode": cell_barcodes}
        save_dir.mkdir(parents=True, exist_ok=True)
        h5_path = save_dir / f"{name}_{shape_name}.h5"

        logger.info(f"Extracting {len(coords_topleft)} cell images...")

        if write_in_tmp_dir:
            with tempfile.TemporaryDirectory() as tmp_dir:
                logger.info(f"Using temp dir: {tmp_dir}")
                h5_tmp_path = os.path.join(tmp_dir, f"{name}_{shape_name}.h5")
                patcher.to_h5(
                    h5_tmp_path,
                    extra_assets=extra_assets,
                )
                shutil.copy(h5_tmp_path, save_dir)
        else:
            patcher.to_h5(
                str(h5_path),
                extra_assets=extra_assets,
            )
