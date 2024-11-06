import math
import os
import random

from PIL import Image
from numpy import ndarray
from torch.utils.data import Dataset, Sampler, DistributedSampler, ConcatDataset
from torchvision.transforms.v2 import Transform


class CustomImageFolder(Dataset):
    def __init__(
        self,
        folder_path: str,
        transform: Transform,
    ) -> None:
        self.folder_path = folder_path
        self.transform = transform
        self.image_file = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.folder_path, self.image_file[idx])
        with open(img_path, "rb") as f:
            img = Image.open(f)
            return self.transform(img)


class CustomBatchSampler(Sampler):
    """
    Yield a mini-batch of indices with samples coming only from the same batch variable.`

    Args:
        batch_array (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
        shuffle (bool): whether or nor shuffling the data
    """

    def __init__(
        self, concat_dataset: ConcatDataset, batch_size: int, shuffle: bool, indices: ndarray | None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = {}

        idx_start = 0
        for i, dataset in enumerate(concat_dataset.datasets):
            self.data[dataset.dataset_name] = list(range(idx_start, concat_dataset.cumulative_sizes[i]))

        self.total = 0
        for size, indexes in self.data.items():
            self.total += math.ceil(len(indexes) / self.batch_size)

        self.n_mini_batch_per_batch = {
            key: math.ceil(len(self.data[key]) / self.batch_size) for key in self.data
        }
        self.batch_order = []
        for key in self.n_mini_batch_per_batch.keys():
            self.batch_order += self.n_mini_batch_per_batch[key] * [key]

    def __iter__(self):
        # Prepare order
        if self.shuffle:
            random.shuffle(self.batch_order)
            for key in self.data.keys():
                random.shuffle(self.data[key])

        # Yield batch
        count_idxs = {key: 0 for key in self.data}
        for b in self.batch_order:
            yield self.data[b][count_idxs[b] : count_idxs[b] + self.batch_size]
            count_idxs[b] += self.batch_size

    def __len__(self):
        return self.total


class DistributedBatchSamplerSameDataset(DistributedSampler):
    """Inspired from https://discuss.pytorch.org/t/using-distributedsampler-in-combination-with-batch-sampler-to-make-sure-batches-have-sentences-of-similar-length/119824/3."""
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=10,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = CustomBatchSampler(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            indices=indices,
        )
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size
