import os
from pathlib import Path

import pandas as pd
from loguru import logger

from hest import iter_hest
from scellst.bench.data_formatting import download_and_prepare_data, process_hest_data
from scellst.cellhest_adapter.cell_hest_data import CellHESTData
from scellst.cellhest_adapter.processing_utils import (
    filter_data,
    fetch_data,
    convert_to_cellst,
    plot_cellst,
    save_gene_names,
    remove_dataset,
)
from scellst.constant import MODELS_DIR, CELL_IMG_DIR, CELL_IMG_STAT_DIR, CELL_EMB_DIR
from scellst.utils.utils import run_moco_script


def download_data(
    path_dataset: Path, organ: str | None = None, ids_to_query: list[str] | None = None , shape_name: str = "cellvit"
) -> None:
    assert (organ is not None) ^ (
        ids_to_query is not None
    ), f"Only one should not be none, got: organ={organ} and ids_to_query={ids_to_query}"
    df = pd.read_csv("data/HEST_v1_1_0.csv")
    df = df.set_index("id")
    df = df.sort_index()
    if ids_to_query:
        logger.info(f"Working with preselected {ids_to_query} slides...")
        df = df.loc[ids_to_query]
    else:
        df = filter_data(df, organ)
        ids_to_query = df.index.tolist()
        logger.info(f"Working with selected {ids_to_query} slides from {organ}...")
    technology = df["st_technology"].tolist()
    fetch_data(str(path_dataset), ids_to_query)
    convert_to_cellst(path_dataset, ids_to_query, technology, shape_name)
    plot_cellst(path_dataset, ids_to_query, technology, shape_name)
    if "xenium" in technology:
        save_gene_names(path_dataset, ids_to_query, technology, organ)


def remove_data_organ(path_dataset: Path, organ: str) -> None:
    df = pd.read_csv("external/HEST/assets/HEST_v1_1_0.csv")
    df = filter_data(df, organ)
    ids_to_remove = df["id"].tolist()
    logger.info(f"Working with {organ} slides...")
    remove_dataset(path_dataset, ids_to_remove)


def run_ssl(
    path_dataset: str,
    organ: str | None,
    ids_to_query: list[str] | None,
    tag: str,
    n_gpus: int,
    n_cpus_per_gpu: int,
) -> None:
    assert (organ is not None) ^ (
        ids_to_query is not None
    ), f"Only one should not be none, got: organ={organ} and ids_to_query={ids_to_query}"
    if ids_to_query:
        logger.info(f"Working with preselected {ids_to_query} slides...")
    else:
        df = pd.read_csv("external/HEST/assets/HEST_v1_1_0.csv")
        df = filter_data(df, organ)
        ids_to_query = df["id"].tolist()
        logger.info(f"Working with selected {ids_to_query} slides from {organ}...")
    run_moco_script(
        tag=tag,
        list_slides=ids_to_query,
        path_dataset=path_dataset,
        n_gpus=n_gpus,
        n_cpus_per_gpu=n_cpus_per_gpu,
    )


def embed_cells(
    path_dataset: Path,
    organ: str | None,
    ids_to_query: list[str] | None,
    tag: str,
    model_name: str,
    normalisation_type: str,
    shape_name: str = "cellvit",
) -> None:
    # Load slide ids to embed
    assert (organ is not None) ^ (
        ids_to_query is not None
    ), f"Only one should not be none, got: organ={organ} and ids_to_query={ids_to_query}"
    if ids_to_query:
        logger.info(f"Working with preselected {ids_to_query} slides...")
    else:
        df = pd.read_csv("external/HEST/assets/HEST_v1_1_0.csv")
        df = filter_data(df, organ)
        ids_to_query = df["id"].tolist()
        logger.info(f"Working with selected {ids_to_query} slides from {organ}...")

    for i, st in enumerate(
            iter_hest(
                hest_dir=str(path_dataset), id_list=ids_to_query, load_transcripts=False
            )
    ):
        logger.info(f"Encoding {ids_to_query[i]}...")
        cst = CellHESTData.from_HESTData(st)
        if not "imagenet" in tag:
            weight_path = os.path.join(MODELS_DIR / "ssl", tag, "moco_model_best.pth.tar")
        else:
            weight_path = tag
        cst.dump_cell_embeddings(
            cell_img_save_dir=path_dataset / CELL_IMG_DIR,
            cell_stat_img_save_dir=path_dataset / CELL_IMG_STAT_DIR,
            normalisation_type=normalisation_type,
            save_dir=path_dataset / CELL_EMB_DIR,
            shape_name=shape_name,
            model_name=model_name,
            weights_path=weight_path,
            tag= f"{tag}_{normalisation_type}",
            write_in_tmp_dir=True,
        )
        if "imagenet" in tag:
            cst.dump_cell_one_hot(
                cell_emb_dir=path_dataset / CELL_EMB_DIR,
                tag=f"{tag}_{normalisation_type}",
            )



if __name__ == "__main__":
    download_data(
        # path_dataset=Path("hest_data"), ids_to_query=["TENX62", "TENX90", "TENX70", "TENX72"]
        path_dataset=Path("hest_data"), ids_to_query=["TENX65"]
    )
