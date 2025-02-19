from pathlib import Path
from typing import Any

import torch

from scellst.bench.bench_dataset import MclSTExpDataset, SlideDataset, SlideGraphDataset
from scellst.dataset.data_handler import VisiumHandler


class MclSTExpVisiumHandler(VisiumHandler):
    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return MclSTExpDataset(adata, data_dir)


class HisToGeneVisiumHandler(VisiumHandler):
    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return SlideDataset(adata, data_dir)


class THItoGeneVisiumHandler(VisiumHandler):
    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return SlideGraphDataset(adata, data_dir)


class IstarGeneVisiumHandler(VisiumHandler):
    def create_dataset(self, adata: Any, data_dir: Path) -> torch.utils.data.Dataset:
        return SlideDataset(adata, data_dir)
