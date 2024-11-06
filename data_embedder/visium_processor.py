import itertools
import logging
import os

import cv2
import numpy as np
import scanpy as sc
from PIL import Image
from matplotlib import pyplot as plt
from pandas import Series
from tqdm.auto import tqdm

from .base_processor import BaseProcessor
from .constants import MAPPING_DICT, COLOR_DICT
from .spatial_transformation import load_adata
from .utils_from_hest import find_pixel_size_from_adata


logger = logging.getLogger(__name__)


class VisiumProcessor(BaseProcessor):
    """
    Processor for Visium images.

    Args:
        input_folder_path: Path to the input folder.
        slide_folder_path: Path to the slide folder.
        annotation_folder_path: Path to the annotation folder.
        img_size: Size of the extracted images.
        radius_ratio: Ratio of the radius.
        model_name: Name of the model.
        model_weights: Weights of the model.
    """
    def __init__(
        self,
        input_folder_path: str,
        slide_folder_path: str,
        annotation_folder_path: str,
        img_size: float = 15,
        radius_ratio: float = 1.0,
        model_name: str = "resnet18",
        model_weights: str = "imagenet",
    ):
        self.input_folder_path = input_folder_path
        self.slide_folder_path = slide_folder_path
        self.annotation_folder_path = annotation_folder_path
        self.adata = load_adata(input_folder_path)
        self.radius_ratio = radius_ratio

        tag = f"{model_name}_{img_size}"
        pixel_size = find_pixel_size_from_adata(self.adata)
        img_size = int(img_size / pixel_size)
        logger.info(f"Image pixel size: {img_size}")

        # Find slide file
        data_name = os.path.basename(input_folder_path)
        slide_names = [
            f for f in os.listdir(slide_folder_path) if f.startswith(data_name)
        ]
        assert len(slide_names) > 0, "No slide file found"
        logger.info(f"Slide name: {slide_names[0]}")
        slide_path = os.path.join(slide_folder_path, slide_names[0])
        self.slide_name = os.path.splitext(slide_names[0])[0]

        # Find annotation file
        annotation_names = [
            f for f in os.listdir(annotation_folder_path) if (f.startswith(data_name) & f.endswith(".csv"))
        ]
        assert len(annotation_names) > 0, "No annotation file found"
        logger.info(f"Annotation name: {annotation_names[0]}")
        annotation_path = os.path.join(annotation_folder_path, annotation_names[0])
        assert os.path.exists(annotation_path), "No annotation files found"

        # Output folder file
        output_folder_path = os.path.join(input_folder_path, "mil")
        os.makedirs(output_folder_path, exist_ok=True)

        # Initiate base processor
        super().__init__(
            annotation_path,
            slide_path,
            output_folder_path,
            img_size,
            tag=tag,
            model_name=model_name,
            model_weights=model_weights,
        )

        # Prepare spot information
        self.slide_dim = self.slide.dimensions
        logger.info(f"Slide dim: {self.slide_dim}")
        self.adata_spatial_coords = self.adata.obsm["spatial_img"]  # format (x, y)

        # Prepare annotations coordinates
        self.true_radius = (
            self.adata.uns["spatial"][next(iter(self.adata.uns["spatial"].keys()))][
                "scalefactors"
            ]["spot_diameter_HEres"]
            // 2
        )
        self.physical_radius = self.true_radius * self.radius_ratio
        logger.info(
            f"Annotation range: "
            f"x: {self.annotations['x_center'].min():.2f} - {self.annotations['x_center'].max():.2f}; "
            f"y: {self.annotations['y_center'].min():.2f} - {self.annotations['y_center'].max():.2f}"
        )
        logger.info(
            f"Spatial adata coordinates range: "
            f"(x, y): {self.adata_spatial_coords.min(axis=0)} - {self.adata_spatial_coords.max(axis=0)}"
        )
        # Prepare output folders
        self.output_spot = "outputs/spots"
        os.makedirs(self.output_spot, exist_ok=True)
        self.output_slide = "outputs/slides"
        os.makedirs(self.output_slide, exist_ok=True)

    def _compute_spot_features(
        self,
    ) -> None:
        """Here we work with transformed coordinates to match adata spatial coordinates."""
        # Count cells and remove spots without cells
        is_cell_in_spot = np.zeros(len(self.annotations), dtype=int)
        n_cell_per_spot = np.zeros(len(self.adata), dtype=int)
        n_cell_type_per_spot = np.zeros(
            (len(self.adata), len(MAPPING_DICT.keys())), dtype=int
        )
        spot_cell_map = {}
        spot_cell_distance = {}
        x = self.annotations["x_center"]
        y = self.annotations["y_center"]
        for i, spot_name in tqdm(
            enumerate(self.adata.obs_names), desc="Prepare spot features"
        ):
            # Test if center of cells are in spot i
            cell_distances = (
                (x - self.adata_spatial_coords[i, 0]) ** 2
                + (y - self.adata_spatial_coords[i, 1]) ** 2
            ) ** 0.5
            bool_series = cell_distances <= self.physical_radius

            # Update cell count
            is_cell_in_spot[bool_series] += 1
            n_cell_per_spot[i] = bool_series.sum()

            # Compute occurrence of each cell type in the columns label
            for cell_type, mapping_key in MAPPING_DICT.items():
                n_cell_type_per_spot[i, mapping_key] = (
                    self.annotations.loc[bool_series, "class"].eq(cell_type).sum()
                )
            # Update spot cell map
            spot_cell_map[spot_name] = list(bool_series[bool_series].index)
            spot_cell_distance[spot_name] = list(
                cell_distances[bool_series].values / self.true_radius
            )

            # Plot spot image
            if i < 10:
                self._plot_spot_image(i, bool_series)

        logger.info(
            f"Number of spots with cells: {(n_cell_per_spot > 0).sum()} / {len(self.adata)}"
        )
        self.adata.obs["with_cell"] = n_cell_per_spot > 0
        self.adata.obs["n_cell"] = n_cell_per_spot
        self.adata.obs[list(MAPPING_DICT.keys())] = n_cell_type_per_spot

        # Create MIL information
        self.adata.uns["MIL"] = {}
        self.adata.uns["MIL"][f"spot_cell_map"] = spot_cell_map
        self.adata.uns["MIL"][f"spot_cell_distance"] = spot_cell_distance

    def _plot_spot_image(self, spot_idx: int, cell_bool: Series) -> None:
        # Create output dir
        output_dir = f"{self.output_spot}/{self.slide_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Extract spot image
        x_c, y_c = self.adata_spatial_coords[spot_idx] - self.physical_radius
        box_size = int(self.physical_radius) * 2
        spot_img = np.array(
            self.slide.read_region(
                location=(int(x_c), int(y_c)),
                level=0,
                size=(box_size, box_size),
            )
        )[:, :, :3]
        copy_spot_img = spot_img.copy()

        # Create mask
        mask_img = np.zeros_like(spot_img).astype(np.uint8)

        # Select annotations
        spot_annotations = self.annotations[cell_bool].copy()
        spot_annotations[["x_min", "x_max", "x_center"]] = (
            spot_annotations[["x_min", "x_max", "x_center"]] - x_c
        )
        spot_annotations[["y_min", "y_max", "y_center"]] = (
            spot_annotations[["y_min", "y_max", "y_center"]] - y_c
        )
        spot_annotations[
            ["x_min", "x_max", "x_center", "y_min", "y_max", "y_center"]
        ] = np.clip(
            spot_annotations[
                ["x_min", "x_max", "x_center", "y_min", "y_max", "y_center"]
            ],
            0,
            box_size,
        )

        # Add annotation to plot
        for i in range(len(spot_annotations)):
            x_center, y_center = (
                int(spot_annotations["x_center"].iloc[i]),
                int(spot_annotations["y_center"].iloc[i]),
            )
            color = COLOR_DICT[spot_annotations["label"].iloc[i]]
            copy_spot_img = cv2.circle(
                copy_spot_img, (x_center, y_center), 20, color, 2
            )
            mask_img = cv2.circle(mask_img, (x_center, y_center), 20, color, 2)

        # Plot and save image
        fig, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(spot_img, interpolation="none")
        ax[1].set_title(f"n cells: {len(spot_annotations)}")
        ax[1].imshow(copy_spot_img, interpolation="none")
        ax[2].imshow(mask_img, interpolation="none")
        center = (
            self.physical_radius,
            self.physical_radius,
        )
        circles = [
            plt.Circle(
                center,
                self.physical_radius,
                color="black",
                linewidth=5,
                linestyle="--",
                fill=False,
            )
            for _ in range(2)
        ]
        ax[0].add_patch(circles[0])
        ax[1].add_patch(circles[1])
        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")
        fig.suptitle(f"Quality threshold: {self.quality_threshold}", y=0.9, fontsize=18)
        fig.savefig(
            os.path.join(
                output_dir, f"qt_lame_{self.quality_threshold}_spot_idx_{spot_idx}.png"
            ),
            transparent=True,
        )
        plt.close()

        # Save some cell images from spot
        for i in range(min(len(spot_annotations), 5)):
            x_min, y_min = (
                int(spot_annotations["x_center"].iloc[i])
                - self.extraction_img_size // 2,
                int(spot_annotations["y_center"].iloc[i])
                - self.extraction_img_size // 2,
            )
            x_max, y_max = (
                int(spot_annotations["x_center"].iloc[i])
                + self.extraction_img_size // 2,
                int(spot_annotations["y_center"].iloc[i])
                + self.extraction_img_size // 2,
            )
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, box_size), min(y_max, box_size)
            Image.fromarray(spot_img).crop((x_min, y_min, x_max, y_max)).save(
                os.path.join(output_dir, f"spot_{spot_idx}_cell_{i}.png"),
                transparent=True,
            )

    def _plot_anndata_obs(self) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        adata_plot = self.adata[self.adata.obs["n_cell"] > 0]
        sc.pl.spatial(adata_plot, color="n_cell", show=False, ax=ax[0])
        adata_plot = self.adata[self.adata.obs["n_cell"] == 0]
        sc.pl.spatial(adata_plot, color="n_cell", show=False, ax=ax[1])
        ax[0].set_title("Spots with cells")
        ax[1].set_title("Spots without cells")
        fig.savefig(
            f"{self.output_slide}/qt_lame_{self.slide_name}_{self.quality_threshold}_n_cells.png"
        )
        plt.close()

    def _save_annadata(
        self,
    ) -> None:
        # Filter spots without cell detected
        self.adata = self.adata[self.adata.obs["with_cell"]]

        cell_in_spot = [v for v in self.adata.uns["MIL"][f"spot_cell_map"].values()]
        cell_in_spot = list(itertools.chain.from_iterable(cell_in_spot))
        self.adata.uns["MIL"]["cell_in_spot"] = cell_in_spot
        self.adata.uns["MIL"]["cell_out_spot"] = list(
            set(np.arange(len(self.annotations))).difference(cell_in_spot)
        )
        self.adata.uns["MIL"]["cell_label"] = self.annotations["class"].values
        self.adata.uns["MIL"]["cell_coordinates"] = self.annotations[
            ["x_center", "y_center"]
        ].values.astype(int)
        self.adata.uns["MIL"]["slide_name"] = self.slide_name
        self.adata.uns["MIL"]["physical_radius"] = (
            self.adata.uns["spatial"][next(iter(self.adata.uns["spatial"].keys()))][
                "scalefactors"
            ]["spot_diameter_HEres"]
            // 2
        )
        self.adata.write_h5ad(
            os.path.join(*[self.input_folder_path, "mil/adata_with_mil.h5ad"])
        )
        logger.info("Anndata saved.")

    def process_image(self) -> None:
        self._compute_spot_features()
        self._plot_anndata_obs()
        self._extract_cell_images()
        self._plot_cell_examples()
        self._create_cell_embeddings()
        self._save_annadata()

        os.remove(self.output_folder_path_img)

