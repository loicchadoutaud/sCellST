import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CellH5HESTDataset(Dataset):
    """Dataset to read ST + H&E from .h5"""

    def __init__(self, h5_path, img_transform=None, chunk_size=1000):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size
        with h5py.File(h5_path, "r") as f:
            self.n_chunks = int(np.ceil(len(f["barcode"]) / chunk_size))

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, "r") as f:
            imgs = f["img"][start_idx:end_idx]
            barcodes = f["barcode"][start_idx:end_idx].flatten().tolist()
            coords = f["coords"][start_idx:end_idx]
            spots = f["spot"][start_idx:end_idx].flatten().tolist()
            if "label" in f.keys():
                labels = f["label"][start_idx:end_idx].flatten()
            else:
                labels = np.full(len(imgs), -1)

        if self.img_transform:
            imgs = torch.stack(
                [self.img_transform(Image.fromarray(img)) for img in imgs]
            )

        output_dict = {
            "img": imgs,
            "barcode": barcodes,
            "coords": coords,
            "spot": spots,
            "label": labels,
        }

        return output_dict
