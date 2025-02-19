import torch
from torch.utils.data import DataLoader, Dataset

from scellst.constant import REGISTRY_KEYS
from scellst.dataset.torch_dataset import EmbeddedMilDataset


def custom_collate(batch: list[dict]) -> dict:
    return {
        REGISTRY_KEYS.X_KEY: torch.cat([data[REGISTRY_KEYS.X_KEY] for data in batch]),
        REGISTRY_KEYS.Y_BAG_KEY: torch.utils.data.default_collate(
            [data[REGISTRY_KEYS.Y_BAG_KEY] for data in batch]
        ),
        REGISTRY_KEYS.Y_INS_KEY: torch.cat(
            [data[REGISTRY_KEYS.Y_INS_KEY] for data in batch]
        ),
        REGISTRY_KEYS.INSTANCE_BAG_IDX_KEY: torch.cat(
            [data[REGISTRY_KEYS.INSTANCE_BAG_IDX_KEY] for data in batch]
        ),
        REGISTRY_KEYS.PTR_BAG_INSTANCE_KEY: torch.unique_consecutive(
            torch.cat([data[REGISTRY_KEYS.INSTANCE_BAG_IDX_KEY] for data in batch]),
            return_inverse=True,
        )[1],
        REGISTRY_KEYS.SIZE_FACTOR: torch.utils.data.default_collate(
            [data[REGISTRY_KEYS.Y_BAG_KEY] for data in batch]
        ),
    }


def create_dataloader_mil_reg(
    dataset: EmbeddedMilDataset,
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
