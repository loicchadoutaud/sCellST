import os
import subprocess
from pathlib import Path
from loguru import logger

from hest.segmentation.cell_segmenters import cell_segmenter_factory

from hest import VisiumReader
from hest.utils import find_first_file_endswith


def download_and_prepare_data(data_id: str, base_url: str, output_dir: str) -> None:
    # Define paths
    data_dir = Path(output_dir) / data_id
    spatial_dir = data_dir / "spatial"
    data_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)

    # Define file names and URLs
    files = [
        f"{data_id}_FFPE-probes_filtered_feature_bc_matrix.h5",
        f"{data_id}_FFPE-probes_image.tiff.gz",
        f"{data_id}_FFPE-probes_scalefactors_json.json.gz",
        f"{data_id}_FFPE-probes_tissue_hires_image.png.gz",
        f"{data_id}_FFPE-probes_tissue_positions_list.csv.gz",
    ]

    # Download files
    for file in files:
        file_path = data_dir / file
        if not file_path.exists():
            url = f"{base_url}/{file}"
            logger.info(f"Downloading {file}...")
            subprocess.run(["wget", "-nv", url, "-P", str(data_dir)], check=True)
        else:
            logger.info(f"{file} already exists, skipping download.")

    # Decompress .gz files
    gz_files = [file for file in files if file.endswith(".gz")]
    for gz_file in gz_files:
        gz_path = data_dir / gz_file
        decompressed_path = gz_path.with_suffix("")  # Removes .gz suffix
        if not decompressed_path.exists():
            logger.info(f"Decompressing {gz_file}...")
            subprocess.run(["gzip", "-d", str(gz_path)], check=True)
        else:
            logger.info(f"{gz_file} already decompressed, skipping.")

    # Move tissue_positions_list.csv to the spatial directory
    tissue_positions_file = (
        data_dir / f"{data_id}_FFPE-probes_tissue_positions_list.csv"
    )
    spatial_positions_file = spatial_dir / "tissue_positions_list.csv"
    logger.info(f"Moving {tissue_positions_file.name} to spatial directory...")
    tissue_positions_file.rename(spatial_positions_file)

    logger.info(f"Data preparation for {data_id} is complete.")


def segment_cells(data_id_path: Path, pixel_size: float, segmenter_name: str) -> str:
    """
    Performs cell segmentation using the specified segmenter and saves the result.
    """
    segmenter = cell_segmenter_factory(segmenter_name)
    return segmenter.segment_cells(
        str(data_id_path / "processed" / "aligned_fullres_HE.tif"),
        "seg",
        pixel_size,
        save_dir=str(data_id_path),
    )


def ensure_output_dirs(output_dir: Path) -> dict:
    """
    Creates necessary output directories and returns their paths.
    """
    subdirs = [
        "st",
        "wsis",
        "thumbnails",
        "metadata",
        "spatial_plots",
        "cellvit_seg",
        "patches",
        "pixel_size_vis",
        "tissue_seg",
    ]
    paths = {name: output_dir / name for name in subdirs}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def move_file(src: Path, dst: Path, file_type: str):
    """
    Moves a file from the source to the destination, with error handling.
    """
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    else:
        logger.warning(f"{file_type} file {src} does not exist!")


def read_and_save(
    path: str,
    save_plots=True,
    pyramidal=True,
    bigtiff=False,
    plot_pxl_size=False,
    save_img=True,
    segment_tissue=False,
):
    """For internal use, determine the appropriate reader based on the raw data path, and
    automatically process the data at that location, then the processed files are dumped
    to processed/

    Args:
        path (str): path of the raw data
        save_plots (bool, optional): whenever to save the spatial plots. Defaults to True.
        pyramidal (bool, optional): whenever to save as pyramidal. Defaults to True.
    """
    print(f"Reading from {path}...")
    st_object = VisiumReader().read(
        find_first_file_endswith(dir=path, suffix="image.tiff"),
        find_first_file_endswith(dir=path, suffix="filtered_feature_bc_matrix.h5"),
        spatial_coord_path=os.path.join(path, "spatial"),
    )
    logger.info("Loaded object:")
    print(st_object)
    if segment_tissue:
        logger.info("Segment tissue")
        st_object.segment_tissue()
    save_path = os.path.join(path, "processed")
    os.makedirs(save_path, exist_ok=True)
    st_object.save(
        save_path,
        pyramidal=pyramidal,
        bigtiff=bigtiff,
        plot_pxl_size=plot_pxl_size,
        save_img=save_img,
    )
    if save_plots:
        st_object.save_spatial_plot(save_path)
    return st_object


def process_hest_data(
    data_path: str, data_id: str, output_dir: str, segmenter_name: str
) -> None:
    """
    Main function to process HEST data: read, segment cells, and organize output.
    """
    data_path = Path(data_path)
    data_id_path = data_path / data_id

    # Step 1: Read and save the spatial transcriptomics data
    st = read_and_save(
        str(data_id_path),
        save_plots=True,
        pyramidal=True,
        bigtiff=False,
        plot_pxl_size=True,
        save_img=True,
        segment_tissue=True,
    )

    # Step 2: Perform cell segmentation
    path_geojson = segment_cells(data_id_path, st.pixel_size, segmenter_name)

    # Step 3: Prepare output directories
    output_dir = Path(output_dir)
    paths = ensure_output_dirs(output_dir)

    # Step 4: Move processed files to output directories
    src_dir = data_id_path / "processed"
    move_file(src_dir / "aligned_adata.h5ad", paths["st"] / f"{data_id}.h5ad", "Adata")
    move_file(
        src_dir / "aligned_fullres_HE.tif", paths["wsis"] / f"{data_id}.tif", "WSI"
    )
    move_file(
        src_dir / "downscaled_fullres.jpeg",
        paths["thumbnails"] / f"{data_id}_downscaled_fullres.jpeg",
        "Thumbnail",
    )
    move_file(
        src_dir / "metrics.json", paths["metadata"] / f"{data_id}.json", "Metadata"
    )
    move_file(
        src_dir / "spatial_plots.png",
        paths["spatial_plots"] / f"{data_id}_spatial_plots.png",
        "Spatial Plots",
    )
    move_file(
        src_dir / "pixel_size_vis.png",
        paths["pixel_size_vis"] / f"{data_id}_pixel_size_vis.png",
        "Pixel Size Plots",
    )
    move_file(
        src_dir / "tissue_seg_vis.jpg",
        paths["tissue_seg"] / f"{data_id}_vis.jpg",
        "Seg Plots",
    )
    move_file(
        src_dir / "tissue_contours.geojson",
        paths["tissue_seg"] / f"{data_id}_contours.geojson",
        "Spatial Plots",
    )
    move_file(
        Path(path_geojson),
        paths["geojson"] / f"{data_id}_cellvit_seg.geojson",
        "GeoJSON",
    )

    # Dump patches
    st.dump_patches(paths["patches"], f"{data_id}")

    logger.info(f"Processing completed for data_id: {data_id}")
