import logging
import os

import numpy as np
import scanpy as sc
from anndata import AnnData
from numpy import ndarray


logger = logging.getLogger(__name__)


def transform_spatial_information(
    adata: AnnData,
    transformation_matrix: ndarray,
) -> None:
    """
    Transform spatial information using a transformation matrix.
    Used to transform spatial coordinates and scale factor when using other H&E images.

    Args:
        adata: Anndata object.
        transformation_matrix: Transformation matrix.
    """
    # Update spatial coordinates
    adata.obsm["spatial"] = adata.obsm["spatial"].astype(int)
    spatial_coords = np.hstack(
        [adata.obsm["spatial"], np.ones((adata.obsm["spatial"].shape[0], 1), dtype=int)]
    )
    spatial_coords = spatial_coords @ transformation_matrix.T
    adata.obsm["spatial_img"] = spatial_coords[:, :2].astype(int)

    # Update scale factor
    adata.uns["spatial"][next(iter(adata.uns["spatial"].keys()))]["scalefactors"][
        "spot_diameter_HEres"
    ] = (
        adata.uns["spatial"][next(iter(adata.uns["spatial"].keys()))]["scalefactors"][
            "spot_diameter_fullres"
        ]
        * transformation_matrix[0, 0]  # scale factor
    )


def load_adata(input_folder_path: str) -> AnnData:
    """ Load anndata from visium file and preprocess spatial information. """
    # Load anndata
    adata = sc.read_visium(input_folder_path)
    adata.var_names_make_unique()
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(int)
    adata = adata[adata.obsm["spatial"][:, 0] > 0]

    # Load transformation file
    list_transformation_files = [
        f for f in os.listdir(input_folder_path) if f.endswith(".npy")
    ]
    if len(list_transformation_files) > 0:
        logger.info("Found transformation file.")
        transformation_matrix = np.load(
            os.path.join(input_folder_path, list_transformation_files[0])
        )
        transform_spatial_information(adata, transformation_matrix)
    else:
        logger.info("No transformation file found.")
        adata.uns["spatial"][next(iter(adata.uns["spatial"].keys()))]["scalefactors"][
            "spot_diameter_HEres"
        ] = adata.uns["spatial"][next(iter(adata.uns["spatial"].keys()))]["scalefactors"][
            "spot_diameter_fullres"
        ]
        adata.obsm["spatial_img"] = adata.obsm["spatial"].copy()
    return adata
