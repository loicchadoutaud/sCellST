from pathlib import Path

import pandas as pd
from loguru import logger

from scellst.bench.data_formatting import download_and_prepare_data, process_hest_data
from scellst.cellhest_adapter.processing_utils import (
    filter_data,
    fetch_data,
    convert_to_cellst,
    plot_cellst,
    save_gene_names,
    remove_dataset,
)
from scellst.dataset.embedding_utils import encode_all_slides
from scellst.utils import run_moco_script


def download_data(
    path_dataset: Path, organ: str | None = None, ids_to_query: list[str] | None = None, with_plot: bool = False
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
    convert_to_cellst(path_dataset, ids_to_query, technology)
    if with_plot:
        plot_cellst(path_dataset, ids_to_query, technology)
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
        df = pd.read_csv("data/HEST_v1_1_0.csv")
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
) -> None:
    assert (organ is not None) ^ (
        ids_to_query is not None
    ), f"Only one should not be none, got: organ={organ} and ids_to_query={ids_to_query}"
    if ids_to_query:
        logger.info(f"Working with preselected {ids_to_query} slides...")
    else:
        df = pd.read_csv("data/HEST_v1_1_0.csv")
        df = filter_data(df, organ)
        ids_to_query = df["id"].tolist()
        logger.info(f"Working with selected {ids_to_query} slides from {organ}...")
    encode_all_slides(
        path_dataset,
        ids_to_query,
        tag,
        model_name=model_name,
        normalisation_type=normalisation_type,
    )


def download_and_prepare_pdac(
    data_id: str, base_url: str, interim_dir: str, output_dir: str
) -> None:
    download_and_prepare_data(
        data_id=data_id,
        base_url=base_url,
        output_dir=interim_dir,
    )
    process_hest_data(
        data_path=interim_dir,
        data_id=data_id,
        output_dir=output_dir,
        segmenter_name="cellvit",
    )


if __name__ == "__main__":
    download_data(
        path_dataset=Path("hest_data"), organ="Ovary", ids_to_query=["TENX39", "TENX65"]
    )
