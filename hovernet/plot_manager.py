import PIL
import cv2
import glob
import json
import numpy as np
import openslide
import os
import pathlib
from matplotlib import pyplot as plt
from typing import List, Tuple

from constants import TYPE_INFO
from mask_utils import make_auto_mask, get_x_y
from misc.utils import log_info
from misc.viz_utils import visualize_instances_dict, visualize_instance_slides
from misc.wsi_handler import OpenSlideHandler


class PlotManager:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_dir = os.path.join(self.output_dir, "plot")
        os.makedirs(self.save_dir, exist_ok=True)

    def _plot_thumbnail(self, wsi_basename: str, save_dir: str) -> None:
        log_info(f"Plotting thumbnail...")

        mask_path_wsi = os.path.join(self.output_dir, "mask", wsi_basename + ".png")
        thumb_path_wsi = os.path.join(self.output_dir, "thumb", wsi_basename + ".png")

        thumb = cv2.cvtColor(cv2.imread(thumb_path_wsi), cv2.COLOR_BGR2RGB)
        self.mask = cv2.cvtColor(cv2.imread(mask_path_wsi), cv2.COLOR_BGR2RGB)

        # plot the low resolution thumbnail along with the tissue mask
        plt.figure(figsize=(15, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(thumb)
        plt.axis("off")
        plt.title("Thumbnail", fontsize=25)

        plt.subplot(1, 2, 2)
        plt.imshow(self.mask)
        plt.axis("off")
        plt.title("Mask", fontsize=25)

        plt.savefig(os.path.join(save_dir, "mask.png"))
        plt.close()

    def _load_results(
        self, wsi_basename: str, wsi_file: str,
    ) -> Tuple[List, List, List, List, int]:
        log_info(f"Loading results...")
        json_path_wsi = os.path.join(self.output_dir, "json",  wsi_basename + ".json")

        bbox_list_wsi = []
        centroid_list_wsi = []
        contour_list_wsi = []
        type_list_wsi = []

        # add results to individual lists
        with open(json_path_wsi) as json_file:
            data = json.load(json_file)
            mag_info = data["mag"]
            nuc_info = data["nuc"]
            for inst in nuc_info:
                inst_info = nuc_info[inst]
                inst_centroid = inst_info["centroid"]
                centroid_list_wsi.append(inst_centroid)
                inst_contour = inst_info["contour"]
                contour_list_wsi.append(inst_contour)
                inst_bbox = inst_info["bbox"]
                bbox_list_wsi.append(inst_bbox)
                inst_type = inst_info["type"]
                type_list_wsi.append(inst_type)

        # load the wsi object and read region
        self.wsi_obj = openslide.OpenSlide(wsi_file)
        return (
            bbox_list_wsi,
            centroid_list_wsi,
            contour_list_wsi,
            type_list_wsi,
            mag_info,
        )

    def _plot_tiles(
        self,
        contour_list_wsi: List,
        type_list_wsi: List,
        save_dir: str,
        w_tile: int = 1000,
        h_tile: int = 1000,
        n_tile: int = 10,
    ) -> None:
        log_info(f"Plotting tile results...")

        mask_level = 4
        mask = make_auto_mask(self.wsi_obj, mask_level=mask_level).T

        for i in range(n_tile):
            # Select tiles at random
            x_indices, y_indices = np.where(mask == 1)

            # Randomly select an index
            selected_index = np.random.choice(len(x_indices), size=1)[0]

            # Get the coordinates of the selected pixel
            x_tile, y_tile = x_indices[selected_index], y_indices[selected_index]
            x_tile, y_tile = get_x_y(self.wsi_obj, (x_tile, y_tile), mask_level, integer=True)
            print(f"Tile {i} coords: {x_tile, y_tile}")
            wsi_tile = self.wsi_obj.read_region(location=(x_tile, y_tile), level=0, size=(w_tile, h_tile))
            wsi_tile = np.array(wsi_tile)[..., :3]

            # only consider results that are within the tile
            coords_xmin = x_tile
            coords_xmax = x_tile + w_tile
            coords_ymin = y_tile
            coords_ymax = y_tile + h_tile

            tile_info_dict = {}
            count = 0
            for idx, cnt in enumerate(contour_list_wsi):
                cnt_tmp = np.array(cnt)
                cnt_tmp = cnt_tmp[
                    (cnt_tmp[:, 0] >= coords_xmin)
                    & (cnt_tmp[:, 0] <= coords_xmax)
                    & (cnt_tmp[:, 1] >= coords_ymin)
                    & (cnt_tmp[:, 1] <= coords_ymax)
                ]
                label = str(type_list_wsi[idx])
                if cnt_tmp.shape[0] > 0:
                    cnt_adj = np.round(cnt_tmp - np.array([x_tile, y_tile])).astype(
                        "int"
                    )
                    tile_info_dict[idx] = {"contour": cnt_adj, "type": label}
                    count += 1

            plt.figure(figsize=(30, 15))

            plt.subplot(1, 2, 1)
            plt.imshow(wsi_tile)
            plt.axis("off")
            plt.title("Input image", fontsize=25)

            plt.subplot(1, 2, 2)
            overlaid_output = visualize_instances_dict(
                wsi_tile, tile_info_dict, type_colour=TYPE_INFO
            )
            plt.imshow(overlaid_output)
            plt.axis("off")
            plt.title("Segmentation Overlay", fontsize=25)

            plt.savefig(os.path.join(save_dir, f"tile_{i}.png"))
            plt.close()

    def _plot_slide(
        self,
        centroid_list_wsi: List,
        type_list_wsi: List,
        save_dir: str,
        level: int = 3,
    ) -> None:
        log_info(f"Plotting slide results...")

        size = self.wsi_obj.level_dimensions[level]
        wsi_slide = self.wsi_obj.read_region((0, 0), level=level, size=size)
        wsi_array = np.array(wsi_slide)[..., :3]

        plt.figure(figsize=(45, 15))

        plt.subplot(1, 3, 1)
        plt.imshow(wsi_array)
        plt.axis("off")
        plt.title("Input image", fontsize=25)

        overlaid_output, black_overlay = visualize_instance_slides(
            wsi_array,
            centroid_list_wsi,
            type_list_wsi,
            type_colour=TYPE_INFO,
            ds_level=level,
        )
        plt.subplot(1, 3, 2)
        plt.imshow(overlaid_output)
        plt.axis("off")
        plt.title("Segmentation Overlay", fontsize=25)

        plt.subplot(1, 3, 3)
        plt.imshow(black_overlay)
        plt.axis("off")
        plt.title("Segmentation Black Overlay", fontsize=25)

        plt.savefig(os.path.join(save_dir, f"slide.png"))
        plt.close()

    def _plot_cells(
        self,
        centroid_list_wsi: List,
        type_list_wsi: List,
        save_dir: str,
        extraction_img_size: int = 48,
        n_cell_per_plot: int = 50,
    ) -> None:
        log_info(f"Plotting cells results...")

        for label in TYPE_INFO.keys():
            idx_cells = np.where(np.asarray(type_list_wsi) == int(label))[0]
            idx_cell_to_plot = np.random.choice(idx_cells, n_cell_per_plot)

            fig, axes = plt.subplots(5, 10, figsize=(50, 25))
            for i, ax in zip(idx_cell_to_plot, axes.flatten()):
                x, y = centroid_list_wsi[i][0] - extraction_img_size // 2, centroid_list_wsi[i][1] - extraction_img_size // 2
                img_cell = self.wsi_obj.read_region((int(x), int(y)), level=0, size=(extraction_img_size, extraction_img_size))
                ax.imshow(img_cell)
                ax.axis('off')
            fig.suptitle(f"Example of cells with label: {TYPE_INFO[label][0]}", fontsize=50, y=1)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"gallery_{TYPE_INFO[label][0]}.png"))
            plt.close(fig)

    def _plot_wsi_output(self, wsi_file: str) -> None:
        wsi_basename = os.path.basename(wsi_file)
        wsi_basename, wsi_ext = os.path.splitext(wsi_basename)

        if os.path.exists(os.path.join(self.output_dir, "csv_annotation",  wsi_basename + '_hovernet_annotations.csv')):
            log_info("Output files found, plotting...")

            save_dir = os.path.join(self.save_dir, wsi_basename)
            os.makedirs(save_dir, exist_ok=True)

            # Plot mask
            self._plot_thumbnail(wsi_basename, save_dir)

            # Load results
            (
                bbox_list_wsi,
                centroid_list_wsi,
                contour_list_wsi,
                type_list_wsi,
                mag_info,
            ) = self._load_results(wsi_basename, wsi_file)

            # Plot slide
            self._plot_slide(centroid_list_wsi, type_list_wsi, save_dir)

            # Plot tiles
            self._plot_tiles(contour_list_wsi, type_list_wsi, save_dir)

            # PLot cells
            self._plot_cells(centroid_list_wsi, type_list_wsi, save_dir)
        else:
            log_info("Output files not found, skipping to next slide...")

    def plot_wsi_list(self):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py

        """
        wsi_path_list = glob.glob(self.input_dir + "/*")
        wsi_path_list.sort()

        for wsi_path in wsi_path_list:
            log_info(f"Plotting slide {wsi_path}...")
            self._plot_wsi_output(wsi_path)
        log_info("End of plotting.")


def get_file_handler(path, backend):
    if backend in [
        ".svs",
        ".tif",
        ".vms",
        ".vmu",
        ".ndpi",
        ".scn",
        ".mrxs",
        ".tiff",
        ".svslide",
        ".bif",
    ]:
        return OpenSlideHandler(path)
    else:
        assert False, "Unknown WSI format `%s`" % backend
