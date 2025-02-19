import os
from pathlib import Path

from loguru import logger

from scellst.cellhest_adapter.cell_hest_data import CellHESTData
from scellst.constant import CELL_IMG_DIR, CELL_EMB_DIR, CELL_IMG_STAT_DIR, MODELS_DIR

from hest import HESTData, iter_hest


def encode_cell_images(
    st: HESTData,
    dataset_path: Path,
    tag: str,
    weight_dir: str,
    model_name: str,
    normalisation_type: str,
) -> None:
    cst = CellHESTData.from_HESTData(st)
    if not "imagenet" in tag:
        weight_path = os.path.join(weight_dir, tag, "moco_model_best.pth.tar")
    else:
        weight_path = tag
    tag += f"_{normalisation_type}"
    cst.dump_cell_embeddings(
        cell_img_save_dir=dataset_path / CELL_IMG_DIR,
        cell_stat_img_save_dir=dataset_path / CELL_IMG_STAT_DIR,
        normalisation_type=normalisation_type,
        save_dir=dataset_path / CELL_EMB_DIR,
        model_name=model_name,
        weights_path=weight_path,
        tag=tag,
        write_in_tmp_dir=True,
    )


def encode_all_slides(
    dataset_path: Path,
    ids_to_query: list[str],
    tag: str,
    model_name: str,
    normalisation_type: str,
) -> None:
    for i, st in enumerate(
        iter_hest(
            hest_dir=str(dataset_path), id_list=ids_to_query, load_transcripts=False
        )
    ):
        logger.info(f"Encoding {ids_to_query[i]}...")
        encode_cell_images(
            st,
            dataset_path=dataset_path,
            tag=tag,
            weight_dir=MODELS_DIR / "ssl",
            model_name=model_name,
            normalisation_type=normalisation_type,
        )
