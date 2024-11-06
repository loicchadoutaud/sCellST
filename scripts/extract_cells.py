import argparse
import os

import h5py
import numpy as np
import openslide
import pandas as pd
from openslide import OpenSlide
from pandas.core.interchange.dataframe_protocol import DataFrame
from tqdm.auto import tqdm


SEED = 0
CELL_SIZE_DICT_15 = {
    "CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_tissue_image": 54,
    "GSM6505133_DonorA_FFPE-probes_image": 68,
    "GSM6505134_DonorB_FFPE-probes_image": 68,
    "GSM6505135_DonorC_FFPE-probes_image": 68,
    "Visium_FFPE_Human_Breast_Cancer_image": 43,
}


def get_std_img_and_save(
    x: float,
    y: float,
    idx: int,
    slide_path: str,
    extraction_img_size: int,
    target_img_size: int,
    output_folder: str,
) -> None:
    x, y = int(x), int(y)
    slide = OpenSlide(slide_path)
    img = slide.read_region(
        location=(
            x - extraction_img_size // 2,
            y - extraction_img_size // 2,
        ),
        level=0,
        size=(extraction_img_size, extraction_img_size),
    )
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img = img.resize((target_img_size, target_img_size))
    img.save(os.path.join(output_folder, f"image_{idx}.jpg"))


def save_cells_h5(
    annotations: DataFrame,
    output_folder_path: str,
    cell_size: int,
    slide: OpenSlide,
    target_cell_size: int,
) -> None:
    # Create h5 dataset
    with h5py.File(output_folder_path, "w") as h5file:
        h5dataset_embedding = h5file.create_dataset(
            "images",
            (len(annotations), target_cell_size, target_cell_size, 3),
            chunks=(1, target_cell_size, target_cell_size, 3),
            dtype=np.uint8,
        )
        h5dataset_label = h5file.create_dataset(
            "labels", len(annotations), dtype=np.uint8
        )

        print(f"Extracting {len(annotations)} cells in {output_folder_path}")
        for i in tqdm(range(len(annotations))):
            # Extract image
            x_min = int(annotations.iloc[i]["x_center"] - cell_size // 2)
            y_min = int(annotations.iloc[i]["y_center"] - cell_size // 2)
            cell_image = slide.read_region(
                (x_min, y_min), 0, (cell_size, cell_size)
            ).convert("RGB")
            cell_image = cell_image.resize((target_cell_size, target_cell_size))

            # Store in h5dataset
            h5dataset_embedding[i] = cell_image
            h5dataset_label[i] = annotations.iloc[i]["label"]


def extract_cells_from_slide(
    data_path: str,
    output_folder_path: str,
    slide_file: str,
    extraction_cell_size: int,
    target_cell_size: int,
):
    # Prepare input / output files
    slide_path = [os.path.join(data_path, "wsi", f) for f in os.listdir(os.path.join(data_path, "wsi")) if f.startswith(slide_file)]
    assert len(slide_path) > 0, f"No slide found in {os.listdir(os.path.join(data_path, 'wsi'))}"
    assert len(slide_path) == 1, f"Multiple slides found in {os.listdir(os.path.join(data_path, 'wsi'))}"
    slide_path = slide_path[0]
    slide_name = os.path.splitext(slide_file)[0]

    # Find annotation file
    annotation_names = [
        f
        for f in os.listdir(os.path.join(data_path, "wsi_out", "csv_annotation"))
        if (f.startswith(slide_name) & f.endswith(".csv"))
    ]
    assert len(annotation_names) > 0, "No slide name found"
    annotation_path = os.path.join(
        data_path, "wsi_out", "csv_annotation", annotation_names[0]
    )
    assert os.path.exists(annotation_path), "No annotation files found"

    # Prepare output folder.
    output_folder = os.path.join(output_folder_path, "cell_images")
    os.makedirs(output_folder, exist_ok=True)

    slide = openslide.OpenSlide(slide_path)
    annotations = pd.read_csv(annotation_path, index_col=0)
    if "label" not in annotations.columns:
        annotations["label"] = "None"

    # Extract cells
    if slide_file.endswith(".tiff"):
        cell_size = CELL_SIZE_DICT_15[slide_name] / 15 * extraction_cell_size
    else:
        mpp = float(slide.properties["openslide.mpp-x"])
        cell_size = extraction_cell_size / mpp
    cell_size = int(cell_size * 1.5)
    print(f"Extracting image of size {cell_size} pixels")

    h5_file = os.path.join(output_folder, slide_name + ".h5")
    save_cells_h5(
        annotations, h5_file, cell_size, slide, target_cell_size
    )
    print(
        f"Size of h5 dataset: {os.path.getsize(h5_file) / (1024**3):.2f} Gb for {len(annotations)}"
    )


if __name__ == "__main__":
    print("Start python script.")
    parser = argparse.ArgumentParser(
        description="Extract cell images based on annotations."
    )
    parser.add_argument("--data_path")
    parser.add_argument("--output_folder_path")
    parser.add_argument("--slide_name", type=str)
    parser.add_argument(
        "--extraction_cell_size",
        default=12,
        type=int,
        help="",
    )
    parser.add_argument(
        "--target_cell_size",
        default=72,
        type=int,
        help="target size of cell images (48 * 1.5)",
    )
    args = parser.parse_args()
    extract_cells_from_slide(
        data_path=args.data_path,
        output_folder_path=args.output_folder_path,
        slide_file=args.slide_name,
        extraction_cell_size=args.extraction_cell_size,
        target_cell_size=args.target_cell_size,
    )
    print("End of python script.")
