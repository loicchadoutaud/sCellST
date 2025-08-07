### Adapted from https://github.com/gustaveroussy/sopa/blob/master/sopa/aggregation/transcripts.py#L20

from functools import partial

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.diagnostics import ProgressBar
from loguru import logger
from scipy.sparse import csr_matrix


def count_transcripts(geo_df: gpd.GeoDataFrame, points: dd.DataFrame, value_key: str) -> AnnData:
    """Count transcripts per cell. The cells and points have to be aligned (i.e., in the same coordinate system)

    Args:
        geo_df: Cells geometries
        points: Transcripts dataframe
        value_key: Key of `points` containing the genes names

    Returns:
        An `AnnData` object of shape `(n_cells, n_genes)` with the counts per cell
    """
    points[value_key] = points[value_key].astype("category").cat.as_known()
    gene_names = points[value_key].cat.categories.astype(str)

    X = csr_matrix((len(geo_df), len(gene_names)), dtype=int)
    adata = AnnData(X=X, var=pd.DataFrame(index=gene_names))
    adata.obs_names = geo_df.index.astype(str)

    geo_df = geo_df.reset_index()

    X_partitions = []

    with ProgressBar():
        points.map_partitions(
            partial(_add_csr, X_partitions, geo_df, gene_column=value_key, gene_names=gene_names),
            meta=(),
        ).compute()

    for X_partition in X_partitions:
        adata.X += X_partition

    logger.info(f"Transcripts in matrix {adata.X.sum()} / {len(points)} ({adata.X.sum() / len(points) * 100:.1f} %)")

    return adata


def _add_csr(
    X_partitions: list[csr_matrix],
    geo_df: gpd.GeoDataFrame,
    partition: pd.DataFrame,
    gene_column: str,
    gene_names: list[str],
) -> None:
    points_gdf = gpd.GeoDataFrame(partition, geometry=gpd.points_from_xy(partition["he_x"], partition["he_y"]))
    joined = geo_df.sjoin(points_gdf)
    cells_indices, column_indices = joined.index, joined[gene_column].cat.codes

    cells_indices = cells_indices[column_indices >= 0]
    column_indices = column_indices[column_indices >= 0]

    X_partition = csr_matrix(
        (np.full(len(cells_indices), 1), (cells_indices, column_indices)),
        shape=(len(geo_df), len(gene_names)),
    )

    X_partitions.append(X_partition)