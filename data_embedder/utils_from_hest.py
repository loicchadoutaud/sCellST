"""Code imported and adapted from https://github.com/mahmoodlab/HEST"""

import os
from enum import Enum
from typing import Tuple

import numpy as np
import openslide
import pandas as pd
import scanpy as sc
from anndata import AnnData
from openslide import OpenSlide
from tifffile import tifffile

from .spatial_transformation import transform_spatial_information


class SpotPacking(Enum):
    """Types of ST spots disposition,
    for Orange Crate Packing see:
    https://kb.10xgenomics.com/hc/en-us/articles/360041426992-Where-can-I-find-the-Space-Ranger-barcode-whitelist-and-their-coordinates-on-the-slide
    """

    ORANGE_CRATE_PACKING = 0
    GRID_PACKING = 1


def find_pixel_size_from_spot_coords(
        my_df: pd.DataFrame,
        inter_spot_dist: float = 100.0,
        packing: SpotPacking = SpotPacking.ORANGE_CRATE_PACKING,
) -> Tuple[float, int]:
    """Estimate the pixel size of an image in um/px given a dataframe containing the spot coordinates in that image

    Args:
        my_df (pd.DataFrame): dataframe containing the coordinates of each spot in an image, it must contain the following columns:
            ['pxl_row_in_fullres', 'pxl_col_in_fullres', 'array_col', 'array_row']
        inter_spot_dist (float, optional): the distance in um between two spots on the same row. Defaults to 100..
        packing (SpotPacking, optional): disposition of the spots on the slide. Defaults to SpotPacking.ORANGE_CRATE_PACKING.

    Raises:
        Exception: if cannot find two spots on the same row

    Returns:
        Tuple[float, int]: approximation of the pixel size in um/px and over how many spots that pixel size was estimated
    """

    def _cart_dist(start_spot, end_spot):
        """cartesian distance in pixel between two spots"""
        d = np.sqrt(
            (start_spot["pxl_col_in_fullres"] - end_spot["pxl_col_in_fullres"]) ** 2
            + (start_spot["pxl_row_in_fullres"] - end_spot["pxl_row_in_fullres"]) ** 2
        )
        return d

    df = my_df.copy()

    max_dist_col = 0
    approx_nb = 0
    best_approx = 0
    df = df.sort_values("array_row")
    for _, row in df.iterrows():
        y = row["array_col"]
        x = row["array_row"]
        if len(df[df["array_row"] == x]) > 1:
            b = df[df["array_row"] == x]["array_col"].idxmax()
            start_spot = row
            end_spot = df.loc[b]
            dist_px = _cart_dist(start_spot, end_spot)

            div = 1 if packing == SpotPacking.GRID_PACKING else 2
            dist_col = abs(df.loc[b, "array_col"] - y) // div

            approx_nb += 1

            if dist_col > max_dist_col:
                max_dist_col = dist_col
                best_approx = inter_spot_dist / (dist_px / dist_col)
            if approx_nb > 3:
                break

    if approx_nb == 0:
        raise Exception("Couldn't find two spots on the same row")

    return best_approx, max_dist_col


def find_pixel_size_from_adata(
    adata: AnnData,
) -> float:
    """Estimate the pixel size of an image in um/px given an anndata with the spot coordinates"""
    data = np.concatenate(
        [adata.obsm["spatial_img"], adata.obs[["array_col", "array_row"]].values], axis=1
    ).astype(int)
    df = pd.DataFrame(
        data=data,
        columns=["pxl_col_in_fullres", "pxl_row_in_fullres", "array_col", "array_row"],
    )
    approx, n = find_pixel_size_from_spot_coords(df)
    return approx


def find_pixel_size_from_HE(slide_path: str) -> float:
    wsi = OpenSlide(slide_path)
    return 1 / (float(wsi.properties["tiff.XResolution"]) * (1.e-4))


def check_pixel_size(adata_path: str) -> None:
    adata = sc.read_visium(adata_path)
    if os.path.exists(os.path.join(adata_path, "transformation_matrix.npy")):
        print("Found transformation matrix.")
        transformation_matrix = np.load(
            os.path.join(adata_path, "transformation_matrix.npy")
        )
        transform_spatial_information(adata, transformation_matrix)
        adata.obsm["spatial"] = adata.obsm["spatial_img"]
    data = np.concatenate(
        [adata.obsm["spatial"], adata.obs[["array_col", "array_row"]].values], axis=1
    ).astype(int)
    df = pd.DataFrame(
        data=data,
        columns=["pxl_col_in_fullres", "pxl_row_in_fullres", "array_col", "array_row"],
    )
    approx, n = find_pixel_size_from_spot_coords(df)
    print(f"Slide: {os.path.basename(adata_path)}")
    print(f"Found {approx:.3f} um/pxl.")
    for cell_image_size in [15]:
        print(f"'{os.path.basename(adata_path)}': {int(cell_image_size / approx)} width cell images.")


def pixel_size_to_mag(pixel_size: float) -> str:
    """ convert pixel size in um/px to a rough magnitude

    Args:
        pixel_size (float): pixel size in um/px

    Returns:
       str: rought magnitude corresponding to the pixel size
    """

    if pixel_size <= 0.1:
        return '60x'
    elif 0.1 < pixel_size and pixel_size <= 0.25:
        return '40x'
    elif 0.25 < pixel_size and pixel_size <= 0.5:
        return '40x'
    elif 0.5 < pixel_size and pixel_size <= 1:
        return '20x'
    elif 1 < pixel_size and pixel_size <= 4:
        return '10x'
    elif 4 < pixel_size:
        return '<10x'


def tiff_save(
        img: np.ndarray, save_path: str, pixel_size: float, pyramidal=True, bigtiff=False
) -> None:
    """Save an image stored in a numpy array to the generic tiff format

    Args:
        img (np.ndarray): image stored in a number array, shape must be H x W x C
        save_path (str): full path to tiff (including filename)
        pixel_size (float): pixel size (in um/px) that will be embedded in the tiff
        pyramidal (bool, optional): whenever to save to a pyramidal format (WARNING saving to a pyramidal format is much slower). Defaults to True.
        bigtiff (bool, optional): whenever to save as a generic BigTiff, must be set to true if the resulting image is more than 4.1 GB . Defaults to False.
    """

    if pyramidal:
        import pyvips
        print("saving to pyramidal tiff... can be slow")
        pyvips_img = pyvips.Image.new_from_array(img)

        # save in the generic tiff format readable by both openslide and QuPath
        pyvips_img.tiffsave(
            save_path,
            bigtiff=bigtiff,
            pyramid=True,
            tile=True,
            tile_width=256,
            tile_height=256,
            compression="deflate",
            resunit=pyvips.enums.ForeignTiffResunit.CM,
            xres=1.0 / (pixel_size * 1e-3),
            yres=1.0 / (pixel_size * 1e-3),
        )
    else:
        with tifffile.TiffWriter(save_path, bigtiff=bigtiff) as tif:
            options = dict(
                tile=(256, 256),
                compression="deflate",
                resolution=(
                    1.0 / (pixel_size * 1e-4),
                    1.0 / (pixel_size * 1e-4),
                    "CENTIMETER",
                ),
            )
            tif.write(img, **options)


def convert_wsi_directory(input_slide_folder_path: str, adata_folder_path: str) -> None:
    output_folder_path = os.path.join(os.path.dirname(input_slide_folder_path), "wsi")
    os.makedirs(output_folder_path, exist_ok=True)
    list_files = os.listdir(adata_folder_path)
    for file in list_files:
        print(f"Converting {file}")
        # Get pixe size from adata
        adata = sc.read_visium(os.path.join(adata_folder_path, file))
        adata.obsm["spatial_img"] = adata.obsm["spatial"].astype(int)
        pixel_size = find_pixel_size_from_adata(adata)
        print(f"Pixel size: {pixel_size:.3f} um/px")
        slide_name = [f for f in os.listdir(input_slide_folder_path) if f.startswith(file)][0]
        if slide_name.endswith(".tif"):
            img_arr = tifffile.imread(os.path.join(input_slide_folder_path, slide_name))
        else:
            slide = openslide.OpenSlide(os.path.join(input_slide_folder_path, slide_name))
            img_arr = np.array(slide.read_region((0, 0), 0, slide.level_dimensions[0]).convert("RGB"))
        slide_name = os.path.splitext(slide_name)[0]
        save_path = os.path.join(output_folder_path, slide_name + ".tiff")
        tiff_save(img_arr, save_path, pixel_size, pyramidal=True)
    print("Done.")