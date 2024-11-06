from typing import Tuple

import numpy as np
import openslide
from openslide import OpenSlide

from numpy import ndarray
from skimage.color import rgb2gray
from skimage.morphology import square, closing, opening
from skimage.filters import threshold_otsu


def make_auto_mask(slide: OpenSlide, mask_level: int) -> ndarray:
    """make_auto_mask. Create a binary mask from a downsampled version
    of a WSI. Uses the Otsu algorithm and a morphological opening.

    :param slide: WSI. Accepted extension *.tiff, *.svs, *ndpi.
    :param mask_level: level of the pyramidal WSI used to create the mask.
    :return mask: ndarray. Binary mask of the WSI. Dimensions are the one of the
    dowsampled image.
    """
    if mask_level < 0:
        mask_level = len(slide.level_dimensions) + mask_level
    slide = openslide.open_slide(slide) if isinstance(slide, str) else slide
    im = slide.read_region((0,0), mask_level, slide.level_dimensions[mask_level])
    im = np.array(im)[:,:,:3]
    im_gray = rgb2gray(im)
    im_gray = clear_border(im_gray, prop=30)
    size = im_gray.shape
    im_gray = im_gray.flatten()
    pixels_int = im_gray[np.logical_and(im_gray > 0.1, im_gray < 0.98)]
    t = threshold_otsu(pixels_int)
    mask = opening(closing(np.logical_and(im_gray<t, im_gray>0.1).reshape(size), footprint=square(32)), footprint=square(32))
    return mask


def clear_border(mask: ndarray, prop: int):
    r, c = mask.shape
    pr, pc = r//prop, c//prop
    mask[:pr, :] = 0
    mask[r-pr:, :] = 0
    mask[:,:pc] = 0
    mask[:,c-pc:] = 0
    return mask


def get_x_y(slide: OpenSlide, point_l: Tuple[int, int], level: int, integer: bool=True):
    """
    Code @PeterNaylor from useful_wsi.
    Given a point point_l = (x_l, y_l) at a certain level. This function
    will return the coordinates associated to level 0 of this point point_0 = (x_0, y_0).
    Args:
        slide : Openslide object from which we extract.
        point_l : A tuple, or tuple like object of size 2 with integers.
        level : Integer, level of the associated point.
        integer : Boolean, by default True. Wether or not to round
                  the output.
    Returns:
        A tuple corresponding to the converted coordinates, point_0.
    """
    x_l, y_l = point_l
    size_x_l = slide.level_dimensions[level][0]
    size_y_l = slide.level_dimensions[level][1]
    size_x_0 = float(slide.level_dimensions[0][0])
    size_y_0 = float(slide.level_dimensions[0][1])

    x_0 = x_l * size_x_0 / size_x_l
    y_0 = y_l * size_y_0 / size_y_l
    if integer:
        point_0 = (int(x_0), int(y_0))
    else:
        point_0 = (x_0, y_0)
    return point_0