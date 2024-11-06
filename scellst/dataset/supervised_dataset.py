import torch
from anndata import AnnData
from torch import Tensor
from torch.utils.data import Dataset

from scellst import REGISTRY_KEYS


class SupervisedDataset(Dataset):
    def __init__(
        self,
        adata: AnnData,
        obsm_key: str = "X_ovarian",
    ) -> None:
        self.X = torch.from_numpy(adata.obsm[obsm_key]).float()
        self.Y = torch.from_numpy(adata.layers["target"]).float()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            REGISTRY_KEYS.OUTPUT_EMBEDDING: self.X[idx],
            REGISTRY_KEYS.Y_INS_KEY: self.Y[idx]
        }

