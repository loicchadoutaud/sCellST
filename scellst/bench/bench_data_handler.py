from scellst.bench.bench_dataset import MclSTExpDataset, SlideDataset, SlideGraphDataset
from scellst.dataset.data_handler import VisiumHandler


class MclSTExpVisiumHandler(VisiumHandler):
    dataset_cls = MclSTExpDataset

class HisToGeneVisiumHandler(VisiumHandler):
    dataset_cls = SlideDataset

class THItoGeneVisiumHandler(VisiumHandler):
    dataset_cls = SlideGraphDataset

class IstarGeneVisiumHandler(VisiumHandler):
    dataset_cls = SlideDataset
