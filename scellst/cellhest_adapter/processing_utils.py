from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from loguru import logger

from scellst.cellhest_adapter.cell_hest_data import CellHESTData, CellXeniumHESTData
from scellst.constant import (
    CELL_IMG_DIR,
    CELL_PLOT_DIR,
    CELL_GENE_DIR,
    DATA_DIR,
    CELL_IMG_STAT_DIR,
)
from scellst.dataset.data_handler import XeniumHandler


from hest import iter_hest, HESTData


def filter_data(df: pd.DataFrame, organ: str) -> pd.DataFrame:
    df = df[df["species"] == "Homo sapiens"]
    df = df[df["st_technology"].isin(["Visium", "Xenium"])]
    df = df[
        (df["preservation_method"] == "FFPE")
        | (df["subseries"].str.contains("ffpe", case=False, na=False))
    ]
    df = df[df["organ"] == organ]
    return df


def load_gene_names(dataset_path: Path, id: str) -> list[str]:
    dataset_handler = XeniumHandler()
    adata = dataset_handler.load_data(dataset_path, id)
    adata = dataset_handler.filter_genes(adata)
    return adata.var_names.tolist()


def fetch_data(path_dataset: str, ids_to_query: pd.Series) -> None:
    logger.info(f"Downloading {len(ids_to_query)} slides...")
    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    datasets.load_dataset(
        "MahmoodLab/hest", cache_dir=path_dataset, patterns=list_patterns, split="all"
    )
    logger.info("Download done.")


def perform_visium_processing(st: HESTData, dataset_path: Path) -> None:
    cst = CellHESTData.from_HESTData(st)
    cst.dump_cell_images(save_dir=dataset_path / CELL_IMG_DIR, write_in_tmp_dir=True)
    cst.dump_cell_image_stats(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_IMG_STAT_DIR,
    )


def perform_xenium_processing(st: HESTData, dataset_path: Path) -> None:
    cst = CellXeniumHESTData.from_XeniumHESTData(dataset_path, st)
    cst.dump_cell_images(
        save_dir=dataset_path / CELL_IMG_DIR,
        shape_name="xenium_nucleus",
        write_in_tmp_dir=True,
    )
    cst.dump_cell_image_stats(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_IMG_STAT_DIR,
    )
    cst.dump_cell_genes(
        cell_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_GENE_DIR,
        write_in_tmp_dir=True,
    )


def plot_visium(st: HESTData, dataset_path: Path) -> None:
    cst = CellHESTData.from_HESTData(st)
    cst.plot_cell_class_gallery(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_PLOT_DIR,
    )
    cst.plot_cell_random_gallery(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_PLOT_DIR,
    )
    cst.plot_cell_visualisation(save_dir=dataset_path / CELL_PLOT_DIR)
    cst.plot_spots_with_number_of_cells(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_PLOT_DIR,
    )
    for spot_index in np.arange(0, 5):
        cst.plot_spot_and_cell(
            cell_img_save_dir=dataset_path / CELL_IMG_DIR,
            save_dir=dataset_path / CELL_PLOT_DIR,
            spot_idx=spot_index,
        )


def plot_xenium(st: HESTData, dataset_path: Path) -> None:
    cst = CellXeniumHESTData.from_XeniumHESTData(dataset_path, st)
    cst.plot_cell_random_gallery(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        save_dir=dataset_path / CELL_PLOT_DIR,
    )


def convert_to_cellst(
    path_dataset: Path, ids_to_query: list[str], technology: list[str]
) -> None:
    for i, st in enumerate(
        iter_hest(
            hest_dir=str(path_dataset), id_list=ids_to_query, load_transcripts=False
        )
    ):
        logger.info(f"Processing {ids_to_query[i]} {technology[i]}...")
        print(st)
        match technology[i]:
            case "Visium":
                perform_visium_processing(st, path_dataset)
            case "Xenium":
                perform_xenium_processing(st, path_dataset)
            case _:
                raise ValueError(f"This should not happen, got {technology[i]}")
    logger.info("End of processing without errors.")


def plot_cellst(
    path_dataset: Path, ids_to_query: list[str], technology: list[str]
) -> None:
    for i, st in enumerate(
        iter_hest(
            hest_dir=str(path_dataset), id_list=ids_to_query, load_transcripts=False
        )
    ):
        logger.info(f"Plotting {ids_to_query[i]} {technology[i]}...")
        match technology[i]:
            case "Visium":
                plot_visium(st, path_dataset)
            case "Xenium":
                plot_xenium(st, path_dataset)
            case _:
                raise ValueError(f"This should not happen, got {technology[i]}")
    logger.info("End of plotting without errors.")


def save_gene_names(
    path_dataset: Path, ids_to_query: list[str], technology: list[str], organ: str
) -> None:
    gene_names = []
    for i in range(len(ids_to_query)):
        match technology[i]:
            case "Xenium":
                logger.info(f"Loading genes for {ids_to_query[i]} {technology[i]}...")
                gene_names.append(load_gene_names(path_dataset, ids_to_query[i]))
            case _:
                pass
    # Compute the union
    intersection = set.union(*(set(lst) for lst in gene_names))
    gene_names = list(intersection)

    # Filter genes
    filtered_genes = [
        gene
        for gene in gene_names
        if not gene.startswith(
            ("BLANK_", "NegControlCodeword_", "NegControlProbe_", "antisense_")
        )
    ]

    # Save results
    pd.Series(filtered_genes, name="gene").sort_values().to_csv(
        DATA_DIR / f"genes_xenium_{organ}.csv"
    )
    logger.info("End of processing without errors.")


def remove_files_with_identifier(root_dir: Path, id: str):
    """
    Recursively remove files containing the specified identifier in their names.

    Args:
        root_dir (Path): The root directory to search as a Path object.
        id (str): The identifier to search for in file names.
    """
    for file_path in root_dir.rglob(
        "*"
    ):  # Recursively iterate through all files and directories
        if file_path.is_file() and id in file_path.name:
            try:
                file_path.unlink()  # Delete the file
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")


def remove_dataset(path_dataset: Path, ids_to_remove: list[str]) -> None:
    for id in ids_to_remove:
        logger.info(f"Removing {id}...")
        remove_files_with_identifier(path_dataset, id)
    logger.info("End of Removing without errors.")
