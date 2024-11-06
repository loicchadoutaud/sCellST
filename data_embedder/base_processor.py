import json
import logging
import os
import re
import uuid
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openslide import OpenSlide
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from .instance_embedder import InstanceEmbedder
from .utils import CustomImageFolderH5

logger = logging.getLogger(__name__)


class BaseProcessor:
    """
    Base class for processing data.

    Args:
        annotation_path: Path to the annotation file.
        slide_path: Path to the slide file.
        output_folder_path: Path to the output folder.
        img_size: Size of the extracted images.
        tag: Tag for the output files.
        target_img_size: Target size of the extracted images.
        use_bbox: Whether to use bounding boxes.
        quality_threshold: Quality threshold for the extracted images.
        model_name: Name of the model.
        model_weights: Weights of the model.
        remove_data: Whether to remove the data.
        copy_data: Whether to copy the data.
    """
    def __init__(
        self,
        annotation_path: str,
        slide_path: str,
        output_folder_path: str,
        img_size: int,
        tag: str,
        target_img_size: int = 48,
        use_bbox: bool = False,
        quality_threshold: float = 0.0,
        model_name: str = "resnet18",
        model_weights: str = "imagenet",
        remove_data: bool = False,
        copy_data: bool = False,
    ):
        self.slide_path = slide_path
        self.annotation_path = annotation_path
        self.extraction_img_size = img_size
        self.target_img_size = target_img_size
        self.use_bbox = use_bbox
        self.quality_threshold = quality_threshold
        self.model_name = model_name
        self.model_weights = model_weights
        self.remove_data = remove_data
        self.copy_data = copy_data

        # Prepare output folders
        self.output_folder_path = output_folder_path
        os.makedirs(self.output_folder_path, exist_ok=True)
        self.output_folder_path_img = os.path.join(
            self.output_folder_path, f"img_{uuid.uuid4()}.h5"
        )
        if "moco" in self.model_weights:
            match = re.search(r'moco_(.*?)_model', self.model_weights)
            if match:
                result = match.group(1)
            else:
                raise ValueError("Not able to process model name")
            model_tag = f"moco_{result}"
        elif "imagenet" in self.model_weights:
            model_tag = "imagenet"
        else:
            model_tag = "other"
        self.output_folder_path_emb = os.path.join(
            self.output_folder_path,
            f"embedding_{model_tag}_{tag}.h5",
        )
        self.output_folders = [self.output_folder_path_img, self.output_folder_path_emb]

        self.slide = OpenSlide(self.slide_path)

        self.annotations = pd.read_csv(self.annotation_path, index_col=0)
        logger.info(f"Found {len(self.annotations)} cells in annotations")
        self.annotations.drop_duplicates(subset=["x_center", "y_center"], inplace=True)
        self.annotations.reset_index(drop=True, inplace=True)

    def _extract_cell_images(self) -> None:
        logger.info("Starting extracting images...")
        with h5py.File(self.output_folder_path_img, "w") as h5file:
            h5dataset = h5file.create_dataset(
                "images",
                (len(self.annotations), self.target_img_size, self.target_img_size, 3),
                dtype=np.uint8
            )
            for i in tqdm(range(len(self.annotations))):
                x, y = int(self.annotations["x_center"].iloc[i]), int(self.annotations["y_center"].iloc[i])
                img = self.slide.read_region(
                    location=(
                        x - self.extraction_img_size // 2,
                        y - self.extraction_img_size // 2,
                    ),
                    level=0,
                    size=(self.extraction_img_size, self.extraction_img_size),
                )
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img = img.resize((self.target_img_size, self.target_img_size))
                h5dataset[i] = img
        logger.info("Image extraction done.")

    def _load_model(self) -> nn.Module:
        assert self.model_name is not None, "No model name provided"
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"Using device {self.device}")
        model = InstanceEmbedder(self.model_name, self.model_weights).to(self.device)
        model.eval()
        return model

    def _load_transform(self) -> Any:
        if "moco" in self.model_weights:
            model_normalisation_path = self.model_weights[:-8] + "_mean_std.json"
            with open(model_normalisation_path, "r") as f:
                norm_dict = json.load(f)
            mean = norm_dict["mean"]
            std = norm_dict["std"]
            logger.info("Loaded moco mean/std normalisation.")
        elif "imagenet" in self.model_weights:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            logger.info("Loaded resnet mean/std normalisation.")
        else:
            raise ValueError("No normalisation available for this weigths.")
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    @torch.inference_mode()
    def _create_cell_embeddings(self) -> None:
        transform = self._load_transform()
        dataset = CustomImageFolderH5(self.output_folder_path_img, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=4096,
            shuffle=False,
            num_workers=6,
            pin_memory=False,
        )
        model = self._load_model()

        # Compute embeddings
        embeddings = np.empty((len(dataset), 2048), dtype=np.float32)
        idx_start = 0
        for batch_img in tqdm(dataloader, "Computing embeddings"):
            batch_img = batch_img.to(self.device)
            embeddings[idx_start : idx_start + len(batch_img)] = (
                model(batch_img).cpu().numpy()
            )
            idx_start += len(batch_img)

        # Save embeddings
        logger.info("Starting saving embeddings...")
        h5file = h5py.File(self.output_folder_path_emb, "w")
        h5file.create_dataset(name="embeddings", shape=embeddings.shape, dtype=embeddings.dtype, data=embeddings)
        logger.info("Emebedding computation done.")

    def _plot_cell_examples(self) -> None:
        fig, axes = plt.subplots(10, 10, figsize=(50, 50))
        idxs = np.random.choice(len(self.annotations), size=100)
        with h5py.File(self.output_folder_path_img, "r") as h5file:
            for i, idx in enumerate(idxs):
                img = h5file["images"][idx]
                if i == 0:
                    logger.info(img.shape)
                axes[i // 10, i % 10].imshow(img)
                axes[i // 10, i % 10].set_title(
                    f"{self.annotations.iloc[idx]['class']}", fontsize=20
                )
                axes[i // 10, i % 10].axis("off")
            fig.suptitle(f"Cell image examples", y=0.9, fontsize=30)
            fig.savefig(
                f"{self.output_folder_path}/cell_examples_{self.use_bbox}_{self.extraction_img_size}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
