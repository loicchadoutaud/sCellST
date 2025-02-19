from pathlib import Path
from typing import NamedTuple

from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDS_DIR = REPORTS_DIR / "predictions"

# Constants
SPOT_DIR = "st"
CELL_IMG_DIR = "cell_images"
CELL_IMG_STAT_DIR = "cell_image_stats"
CELL_EMB_DIR = "cell_embeddings"
CELL_GENE_DIR = "cell_genes"
CELL_PLOT_DIR = "cell_plots"

# Dictionary to map cell type to color
COLOR_MAP = {
    "Connective": (0.30196078431372547, 0.6862745098039216, 0.2901960784313726),
    "Dead": (0.596078431372549, 0.3058823529411765, 0.6392156862745098),
    "Epithelial": (1.0, 0.4980392156862745, 0.0),
    "Inflammatory": (0.21568627450980393, 0.49411764705882355, 0.7215686274509804),
    "Neoplastic": (0.8941176470588236, 0.10196078431372549, 0.10980392156862745),
    "Nolabel": (1, 1, 1),
}

CLASS_LABELS = {
    "Connective": 0,
    "Dead": 1,
    "Epithelial": 2,
    "Inflammatory": 3,
    "Neoplastic": 4,
    "Nolabel": -1,
}
SUB_CLASS_LABELS = ["Connective", "Inflammatory", "Neoplastic"]
REV_CLASS_LABELS = {v: k for k, v in CLASS_LABELS.items()}

# Registry keys
class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    Y_BAG_KEY: str = "Y_bag"
    Y_INS_KEY: str = "Y_ins"
    PTR_BAG_INSTANCE_KEY: str = "ptr_idx_bag_inst"
    INSTANCE_BAG_IDX_KEY: str = "instance_bag_idx"
    OUTPUT_PREDICTION: str = "output_prediction"
    OUTPUT_PROB: str = "output_probability"
    OUTPUT_EMBEDDING: str = "output_embedding"
    OUTPUT_GENE_PARAM: str = "output_gene_param"
    OUTPUT_DISTRIBUTION: str = "output_distribution"
    SIZE_FACTOR: str = "size_factor"


REGISTRY_KEYS = _REGISTRY_KEYS_NT()
