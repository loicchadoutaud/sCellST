import os
import shutil
import tempfile
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import coo_matrix, csr_matrix


from scellst.cellhest_adapter.data_mixin import DataMixin
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
            transcripts_path, columns=["cell_id", "feature_name", "qv"]
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

    def dump_cell_genes(
        self,
        cell_save_dir: str,
        save_dir: str,
        name: str | None = None,
        write_in_tmp_dir: bool = False,
    ):
        if name is None:
            name = self.meta["id"]

        # Load cell barcode
        cell_image_path = os.path.join(cell_save_dir, name + ".h5")
        assert os.path.exists(
            cell_image_path
        ), f"Cell image file not found: {cell_image_path}"
        with h5py.File(cell_image_path, "r") as f:
            cell_ids = f["barcode"][:].flatten().astype(str).astype("object")

        # Get cell gene expression
        df = self.transcript_df[self.transcript_df["cell_id"] != "UNASSIGNED"].copy()

        # Filter low quality transcripts
        n_transcripts = len(df)
        df = df[df["qv"] > 20]
        n_final_transcripts = len(df)
        logger.info(
            f"Kept {n_final_transcripts / n_transcripts * 100: .3f}% high quality transcripts"
        )
        logger.info(f"Kept {len(df['cell_id'].unique())} / {len(cell_ids)} cells")

        df.sort_values(by=["cell_id", "feature_name"], inplace=True)
        long_count_matrix = (
            df.groupby(by=["cell_id", "feature_name"]).size().reset_index(name="counts")
        )

        # Map identifiers to unique indices
        cell_id_to_index = {
            cell_id: idx
            for idx, cell_id in enumerate(long_count_matrix["cell_id"].unique())
        }
        feature_name_to_index = {
            feature_name: idx
            for idx, feature_name in enumerate(
                long_count_matrix["feature_name"].unique()
            )
        }

        # Create sparse matrix
        row = long_count_matrix["cell_id"].map(cell_id_to_index).values
        col = long_count_matrix["feature_name"].map(feature_name_to_index).values
        data = long_count_matrix["counts"].values
        sparse_matrix = coo_matrix(
            (data, (row, col)),
            shape=(len(cell_id_to_index), len(feature_name_to_index)),
        )

        # Convert sparse matrix to CSR format (preferred for AnnData)
        sparse_matrix = sparse_matrix.tocsr()

        # Create AnnData object
        adata = ad.AnnData(
            X=sparse_matrix,  # Gene expression data
            obs=pd.DataFrame(index=list(cell_id_to_index.keys())),  # Cell metadata
            var=pd.DataFrame(index=list(feature_name_to_index.keys())),  # Gene metadata
        )

        # Make sure to match image/embedding data
        cell_id_set = set(cell_ids)  # All desired cell IDs
        current_cell_ids = set(adata.obs_names)  # Existing cell IDs in adata
        missing_cell_ids = cell_id_set - current_cell_ids  # Find missing IDs

        # Log the details
        logger.info(
            f"Found {len(current_cell_ids & cell_id_set)} / {len(cell_id_set)} cells present in the dataset."
        )
        logger.info(
            f"{len(missing_cell_ids)} cells are missing and will be added with zero expression."
        )

        # Add missing cells with zero expression
        if missing_cell_ids:
            # Create a zero matrix for missing cells
            n_genes = adata.shape[1]  # Number of genes (columns)
            zero_matrix = csr_matrix(
                (len(missing_cell_ids), n_genes)
            )  # Sparse matrix of zeros

            # Create `obs` DataFrame for missing cells
            missing_obs = pd.DataFrame(
                index=list(missing_cell_ids)
            )  # Minimal metadata for missing cells

            # Combine existing AnnData with the new cells
            adata_missing = ad.AnnData(
                X=zero_matrix, obs=missing_obs, var=adata.var.copy()
            )
            adata = ad.concat([adata, adata_missing], axis=0, merge="same")

        # Reindex to keep only cells in `cell_ids` (order preserved)
        adata = adata[cell_ids].copy()
        logger.info(
            f"Final dataset contains {adata.shape[0]} cells and {adata.shape[1]} genes."
        )

        # Add spatial coordinates
        gdf = self.get_shapes("xenium_nucleus", "he").shapes
        gdf.index = gdf.index.astype(str)
        gdf = gdf.loc[adata.obs_names]
        adata.obsm["spatial"] = np.stack(
            [gdf.centroid.x, gdf.centroid.y], axis=1
        ).astype(int)

        # Add downsampled image
        register_downscale_img(adata, self.wsi, self.pixel_size)

        # Inspect the AnnData object
        logger.info(adata)

        # Save the AnnData object
        os.makedirs(save_dir, exist_ok=True)
        if write_in_tmp_dir:
            with tempfile.TemporaryDirectory() as tmp_dir:
                logger.info(f"Using temp dir: {tmp_dir}")
                tmp_path = os.path.join(tmp_dir, name + ".h5ad")
                adata.write_h5ad(tmp_path)
                shutil.copy(tmp_path, save_dir)
        else:
            save_path = os.path.join(save_dir, name + ".h5ad")
            adata.write_h5ad(save_path)
