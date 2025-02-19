import os
from abc import ABC
from pathlib import Path

from anndata import AnnData
from loguru import logger

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import random

import cv2
import pandas as pd
import numpy as np
import squidpy as sq
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms


def pad_image_to_max_size(image: Image, max_size: int) -> Image:
    """Pads an image with zeros to make it 224x224 without resizing."""
    # Get the current dimensions
    h, w = image.shape[:2]

    # Calculate padding for height and width
    pad_h = max(0, max_size - h)
    pad_w = max(0, max_size - w)

    # Add padding equally on both sides
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad the image
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return padded_image


class BenchDataset(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        adata: AnnData,
        data_dir: Path,
        train: bool = True,
    ):
        self.train = train
        self.image_path = data_dir / "wsis" / f"{adata.uns['hest_id']}.tif"
        self.r = (
            int(adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"])
            // 2
        )
        logger.info(f"r: {self.r}")
        assert os.path.exists(
            self.image_path
        ), f"Image file not found: {self.image_path}"


class MclSTExpDataset(BenchDataset):
    """Inspired from MclSTExp TenxDataset dataset"""

    def __init__(
        self,
        adata: AnnData,
        data_dir: Path,
        train: bool = True,
    ):
        super().__init__(adata, data_dir, train)
        self.img = cv2.imread(str(self.image_path))
        print(f"image shape: {self.img.shape}")

        self.spatial = pd.DataFrame(
            adata.obsm["spatial"].astype(int), index=adata.obs_names, columns=["x", "y"]
        )
        print(f"spatial coordinate: {self.spatial.max(axis=0)}")
        self.barcodes = adata.obs_names.tolist()
        self.reduced_matrix = adata.X
        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcodes[idx]
        y = self.spatial.loc[self.spatial.index == barcode, "y"].values[0]
        x = self.spatial.loc[self.spatial.index == barcode, "x"].values[0]
        image = self.img[(y - self.r) : (y + self.r), (x - self.r) : (x + self.r)]
        if image.shape[0] != (2 * self.r) or image.shape[1] != (2 * self.r):
            logger.warning("Found too small image.")
            image = pad_image_to_max_size(image, 2 * self.r)

        if self.train:
            image = self.transform(image)

        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["expression"] = torch.tensor(self.reduced_matrix[idx, :]).float()
        item["barcode"] = barcode
        item["position"] = torch.Tensor([x, y])

        return item


class SlideDataset(BenchDataset):
    """Adapated from VITSkin original dataset"""

    def __init__(
        self,
        adata: AnnData,
        data_dir: Path,
        train: bool = True,
    ):
        super().__init__(adata, data_dir, train)
        self.r = 56
        logger.info("Overriding r.")
        self.gene_list = adata.var_names.tolist()

        self.transforms = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
            ]
        )

        self.exps = adata.X
        self.centers = adata.obsm["spatial"].astype(int)
        self.loc = adata.obs[["array_col", "array_row"]].values
        print(self.loc.min(axis=0))
        print(self.loc.max(axis=0))
        print(self.centers.max(axis=0))
        self.patch_dim = 3 * self.r * self.r * 4

        logger.info("Loading imgs...")
        self.img = np.array(Image.open(self.image_path))
        self.positions = torch.LongTensor(self.loc)

    def __getitem__(self, index):
        # Image
        n_patches = len(self.centers)
        patches = torch.zeros((n_patches, self.patch_dim))
        for i in range(n_patches):
            center = self.centers[i]
            y, x = center
            patch = self.img[(x - self.r) : (x + self.r), (y - self.r) : (y + self.r)]
            patch = torch.from_numpy(patch).permute(2, 0, 1)
            # patch = ToTensor()(patch)
            patches[i] = patch.flatten()

        if self.train:
            return patches, self.positions, self.exps
        else:
            return patches, self.positions, self.exps, torch.Tensor(self.centers)

    def __len__(self):
        return 1


class SlideGraphDataset(BenchDataset):
    """Adapated from VITSkin original dataset from HGGEP"""

    def __init__(
        self,
        adata: AnnData,
        data_dir: Path,
        train: bool = True,
    ):
        super().__init__(adata, data_dir, train)
        self.r = 56
        logger.info("Overriding r.")
        self.gene_list = adata.var_names.tolist()

        self.exps = adata.X
        self.centers = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
        self.loc = adata.obs[["array_col", "array_row"]].values
        sq.gr.spatial_neighbors(adata, n_rings=1, coord_type="grid", n_neighs=6)
        self.adj = adata.obsp["spatial_connectivities"].toarray()

        logger.info("Loading imgs...")
        self.img = np.array(Image.open(self.image_path))

    def __getitem__(self, index):
        # Position
        loc = self.loc
        positions = torch.LongTensor(loc)
        centers = torch.LongTensor(self.centers)

        # Expression
        exps = self.exps

        # Image
        n_patches = len(centers)
        patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
        for i in range(n_patches):
            center = centers[i]
            x, y = center
            patch = self.img[(x - self.r) : (x + self.r), (y - self.r) : (y + self.r)]
            patch = torch.from_numpy(patch).permute(2, 0, 1)
            patches[i] = patch

        # Adjacency matrix
        adj = self.adj

        # # Counts values
        # ori, sf = torch.Tensor(self.ori), torch.Tensor(self.sf)

        if self.train:
            return patches, positions, exps, adj
        else:
            return patches, positions, exps, torch.Tensor(centers), adj

    def __len__(self):
        return 1
