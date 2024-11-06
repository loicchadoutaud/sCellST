import logging
import math
import random

import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from scellst import REGISTRY_KEYS
from scellst.dataset.mil_dataset import BaseDataset
from scellst.type import TaskType

logger = logging.getLogger(__name__)


def custom_collate(batch: list[dict]) -> dict:
    return {
        REGISTRY_KEYS.X_KEY: torch.cat([data[REGISTRY_KEYS.X_KEY] for data in batch]),
        REGISTRY_KEYS.Y_BAG_KEY: torch.utils.data.default_collate(
            [data[REGISTRY_KEYS.Y_BAG_KEY] for data in batch]
        ),
        REGISTRY_KEYS.LIBRARY_KEY: torch.utils.data.default_collate(
            [data[REGISTRY_KEYS.LIBRARY_KEY] for data in batch]
        ),
        REGISTRY_KEYS.Y_INS_KEY: torch.cat(
            [data[REGISTRY_KEYS.Y_INS_KEY] for data in batch]
        ),
        REGISTRY_KEYS.BAG_IDX_KEY: torch.utils.data.default_collate(
            [data[REGISTRY_KEYS.BAG_IDX_KEY] for data in batch]
        ),
        REGISTRY_KEYS.BATCH_BAG_IDX_KEY: torch.utils.data.default_collate(
            [data[REGISTRY_KEYS.BATCH_BAG_IDX_KEY] for data in batch]
        ),
        REGISTRY_KEYS.INS_IDX_KEY: torch.cat(
            [data[REGISTRY_KEYS.INS_IDX_KEY] for data in batch]
        ),
        REGISTRY_KEYS.BATCH_INS_IDX_KEY: torch.cat(
            [data[REGISTRY_KEYS.BATCH_INS_IDX_KEY] for data in batch]
        ),
        REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY: torch.unique_consecutive(
            torch.cat([data[REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY] for data in batch]),
            return_inverse=True,
        )[1],
    }

def create_dataloader_mil_reg(
    dataset: BaseDataset,
    batch_size: int,
    train_mode: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_mode,
        num_workers=4,
        collate_fn=custom_collate,
    )


def create_dataloader_mil_instance(
    dataset: Dataset,
    batch_size: int = 4096,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=4, pin_memory=False, shuffle=False
    )


create_dataloader_func_dict = {
    TaskType.regression: create_dataloader_mil_reg,
    TaskType.nb_total_regression: create_dataloader_mil_reg,
    TaskType.nb_mean_regression: create_dataloader_mil_reg,
}
