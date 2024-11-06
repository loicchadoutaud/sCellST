import numpy as np
from typing import NamedTuple

BATCH_SIZE_INFERENCE = 2048
NUM_WORKERS = 4
CELL_TYPE_LIST = [
    "dead cells",
    "epithelial",
    "inflammatory",
    "connective-soft tissue cells",
    "neoplastic cells",
]
EPSILON = 1.0e-13
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]
MAPPING_DICT = {
    "connec": 0,
    "inflam": 1,
    "necros": 2,
    "neopla": 3,
    "no-neo": 4,
    "nolabe": 5,
}
REVERSE_MAPPING_DICT = {v: k for k, v in MAPPING_DICT.items()}

PATCH_SIZE = 1024
N_PATCH_PER_SLIDE = 5
SHORT_METRIC_NAMES = {
    "roc_auc_score": "AUC",
    "average_precision_score": "AP",
    "balanced_accuracy_score": "BAcc",
    "recall": "Rec",
    "precision": "Pre",
}

class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    Y_BAG_KEY: str = "Y_bag"
    Y_INS_KEY: str = "Y_ins"
    LIBRARY_KEY: str = "library"
    BAG_IDX_KEY: str = "bag_idx"
    INS_IDX_KEY: str = "ins_idx"
    PTR_BAG_INSTANCE_KEY: str = "idx_bag_inst"
    BATCH_BAG_IDX_KEY: str = "batch_bag_idx"
    BATCH_INS_IDX_KEY: str = "batch_instance_idx"
    OUTPUT_PREDICTION: str = "output_prediction"
    OUTPUT_SCORE: str = "output_score"
    OUTPUT_DISTRIBUTION: str = "output_distribution"
    OUTPUT_GENE_PARAM: str = "output_gene_param"
    OUTPUT_EMBEDDING: str = "output_embedding"
    OUTPUT_LATENT: str = "output_latent"

REGISTRY_KEYS = _REGISTRY_KEYS_NT()

