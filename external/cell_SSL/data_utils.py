import math
import random

import numpy as np
import torch
from torch.utils.data import Sampler, DistributedSampler, ConcatDataset, Subset
from numpy import ndarray


class CustomBatchSampler(Sampler):
    def __init__(self, dataset: ConcatDataset | Subset, batch_size: int, shuffle: bool = False, drop_last: bool = False, seed: int = 0, epoch: int = 0):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data: dict[str, ndarray] = {}
        self.batch_order = []
        self.seed = seed
        self.epoch = epoch

        # Precompute dataset indices
        idx_start = 0
        if isinstance(dataset, ConcatDataset):
            for i, sub_dataset in enumerate(dataset.datasets):
                dataset_name = getattr(sub_dataset, 'dataset_name', f'dataset_{i}')
                self.data[dataset_name] = np.arange(idx_start, dataset.cumulative_sizes[i])
                idx_start = dataset.cumulative_sizes[i]
                print(f"Precomputed indices for {dataset_name}: {self.data[dataset_name][:10]}...")  # Print first 10 indices for debug
        elif isinstance(dataset, Subset):
            if not isinstance(dataset.dataset, ConcatDataset):
                raise ValueError("For a Subset, the wrapped dataset must be a ConcatDataset.")
            concat_dataset = dataset.dataset
            subset_indices = np.array(dataset.indices)

            for i, sub_dataset in enumerate(concat_dataset.datasets):
                dataset_name = getattr(sub_dataset, 'dataset_name', f'dataset_{i}')
                dataset_range = np.arange(idx_start, concat_dataset.cumulative_sizes[i])
                mask = np.isin(subset_indices, dataset_range, assume_unique=True)
                self.data[dataset_name] = np.where(mask)[0]
                idx_start = concat_dataset.cumulative_sizes[i]
                print(f"Precomputed indices for {dataset_name}: {self.data[dataset_name][:10]}...")  # Print first 10 indices for debug
        else:
            raise ValueError(f"Dataset must be either a ConcatDataset or a Subset object, got {type(dataset)}")

        # Precompute batch counts
        self.n_mini_batch_per_batch = {
            key: len(indices) // batch_size if drop_last else math.ceil(len(indices) / batch_size)
            for key, indices in self.data.items()
        }
        self.batch_order = np.asarray(sum(([key] * count for key, count in self.n_mini_batch_per_batch.items()), []))
        self.total = sum(self.n_mini_batch_per_batch.values())

        print(f"Batch counts per dataset: {self.n_mini_batch_per_batch}")
        print(f"Total number of batches: {self.total}")

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batch_order), generator=g).tolist()
            self.batch_order = self.batch_order[indices]
            for key in self.data.keys():
                indices = torch.randperm(len(self.data[key]), generator=g).tolist()
                self.data[key] = self.data[key][indices]
            print(self.batch_order[:10])

        count_idxs = {key: 0 for key in self.data}
        for key in self.batch_order:
            start_idx = count_idxs[key]
            end_idx = start_idx + self.batch_size

            if self.drop_last and end_idx > len(self.data[key]):
                print(f"Skipping batch for {key} due to drop_last.")
                continue

            yield self.data[key][start_idx:end_idx]
            count_idxs[key] += self.batch_size

    def __len__(self):
        return self.total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



class DistributedBatchSampler(CustomBatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers.

    Args:
        batch_sampler (torch.utils.data.sampler.BatchSampler)
        num_replicas (int, optional): Number of processes participating in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.

    Example:
        >>> from torch.utils.data.sampler import BatchSampler
        >>> from torch.utils.data.sampler import SequentialSampler
        >>> sampler = SequentialSampler(list(range(12)))
        >>> batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        >>>
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=0))
        [[0, 2], [4, 6], [8, 10]]
        >>> list(DistributedBatchSampler(batch_sampler, num_replicas=2, rank=1))
        [[1, 3], [5, 7], [9, 11]]
    """

    def __init__(self, batch_sampler, **kwargs):
        self.batch_sampler = batch_sampler
        self.kwargs = kwargs

    def __iter__(self):
        for batch in self.batch_sampler:
            yield list(DistributedSampler(batch, **self.kwargs))

    def __len__(self):
        return len(self.batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.batch_sampler.set_epoch(epoch)