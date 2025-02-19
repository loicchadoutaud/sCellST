import math
import random

from loguru import logger
from numpy import ndarray
from torch.utils.data import Sampler


class CustomBatchSampler(Sampler):
    """
    Yield a mini-batch of indices with samples coming only from the same batch variable.`

    Args:
        batch_array (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
        shuffle (bool): whether or nor shuffling the data
    """

    def __init__(self, batch_array: ndarray, batch_size: int, shuffle: bool):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = {}
        for i in range(len(batch_array)):
            if batch_array[i] in self.data:
                self.data[batch_array[i]].append(i)
            else:
                self.data[batch_array[i]] = [i]

        self.total = 0
        for size, indexes in self.data.items():
            self.total += math.ceil(len(indexes) / self.batch_size)

        self.n_mini_batch_per_batch = {
            key: math.ceil(len(self.data[key]) / self.batch_size) for key in self.data
        }
        logger.info(self.n_mini_batch_per_batch)
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
