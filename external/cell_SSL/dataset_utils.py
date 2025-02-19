import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.transforms.v2 import Transform, Compose, CenterCrop
from tqdm.auto import tqdm

from moco.loader import TwoCropsTransform


class CustomImageFolderH5(Dataset):
    def __init__(
            self,
            folder_path: str,
            transform: Transform,
    ) -> None:
        self.folder_path = folder_path
        self.transform = transform
        self.dataset_name = os.path.splitext(os.path.basename(folder_path))[0]
        with h5py.File(self.folder_path, 'r') as h5file:
            self.length = h5file["img"].shape[0]

    def _open_hdf5(self) -> None:
        self._h5file = h5py.File(self.folder_path, 'r')
        self._dataset = self._h5file['img']

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        if not hasattr(self, "_h5file"):
            self._open_hdf5()
        return self.transform(Image.fromarray(self._dataset[idx])), self.dataset_name


def create_dataset(
    data_path: str, list_slide_ids: list[str], transform: Transform | Compose
) -> ConcatDataset:
    """Create a single dataset with multiple slides."""
    all_datasets = []
    for slide in list_slide_ids:
        print(f"Creating dataset for slide {slide}...")
        all_datasets.append(
            CustomImageFolderH5(os.path.join(data_path, "cell_images", slide)+".h5", transform)
        )
    print("Merging datasets...")
    return ConcatDataset(all_datasets)


def create_normalized_dataset(
    data_path: str, list_slide_ids: list[str], template_transform: list[Transform]
) -> ConcatDataset:
    """Create a single dataset with multiple slides."""
    all_datasets = []
    for slide in list_slide_ids:
        print(f"Creating dataset for slide {slide}...")
        stats_path = os.path.join(data_path, "cell_image_stats", slide) + ".json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        normalize_transform = transforms.Normalize(stats["mean"], stats["std"])
        transform = TwoCropsTransform(
            transforms.Compose(template_transform + [normalize_transform]), transforms.Compose(template_transform + [normalize_transform])
        )
        all_datasets.append(
            CustomImageFolderH5(os.path.join(data_path, "cell_images", slide)+".h5", transform)
        )
    print("Merging datasets...")
    return ConcatDataset(all_datasets)


def calculate_mean_std(data_path: str, list_slides: list[str], workers: int, n_cell_max: int | None = None) -> tuple[list, list]:
    """Calculate mean and std of the dataset"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            CenterCrop(48),
        ]
    )
    dataset = create_dataset(data_path, list_slides, transform)
    if n_cell_max is not None and (len(dataset) > n_cell_max):
        indices = np.random.choice(len(dataset), n_cell_max, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using only {n_cell_max} cells for training.")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2048, shuffle=False, num_workers=workers, pin_memory=False
    )

    print("Calculating mean and std of the dataset...")
    mean = 0
    std = 0
    for batch, _ in tqdm(dataloader):
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    return mean.tolist(), std.tolist()


def compute_and_save_mean_std(args) -> None:
    # calculate mean and std of the dataset
    mean, std = calculate_mean_std(args.data_path, args.list_slides, args.workers, args.n_cell_max)
    stat_dict = {"mean": mean, "std": std}
    print(stat_dict)
    save_dir = os.path.join(args.output_path, args.tag)
    os.makedirs(save_dir, exist_ok=True)
    with open(
        os.path.join(save_dir, "moco_model_best_mean_std.json"), "w"
    ) as f:
        json.dump(stat_dict, f, indent=4)
        f.truncate()
    print("Mean / std computations done.")