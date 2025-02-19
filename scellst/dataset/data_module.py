from pathlib import Path

import anndata as ad
import lightning as L
import torch
from anndata import AnnData
from loguru import logger
from omegaconf import DictConfig
from scipy.sparse import issparse
from sklearn.preprocessing import RobustScaler
from torch.utils.data import random_split, DataLoader, ConcatDataset

from scellst.bench.bench_data_handler import (
    MclSTExpVisiumHandler,
    HisToGeneVisiumHandler,
    IstarGeneVisiumHandler,
    THItoGeneVisiumHandler,
)
from scellst.dataset.data_handler import (
    XeniumHandler,
    VisiumHandler,
    MilVisiumHandler,
)
from scellst.dataset.dataset_utils import custom_collate
from scellst.dataset.gene_utils import select_labels
from scellst.type import TaskType
from simulation.sim_data_utils import SupervisedInstanceHandler


class STDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        list_training_ids: list[str],
        genes: str | list[str],
        embedding_tag: str,
        dataset_handler: VisiumHandler | XeniumHandler,
        predict_id: str | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        filter_genes: bool = True,
        filter_cells: bool = True,
        normalize: bool = True,
        log1p: bool = True,
        scale: str = "no_scaling",
        frac_train: float = 0.8,
        fold: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.list_training_ids = list_training_ids
        self.predict_id = predict_id
        self.genes = genes
        self.embedding_tag = embedding_tag
        self.dataset_handler = dataset_handler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter_genes = filter_genes
        self.filter_cells = filter_cells
        self.normalize = normalize
        self.log1p = log1p
        self.scale = scale
        self.frac_train = frac_train
        self.fold = fold
        self.gen = torch.Generator().manual_seed(seed)
        if type(dataset_handler) is MilVisiumHandler:
            self.custom_collate = custom_collate
        else:
            self.custom_collate = None

        # Validate IDs
        self._validate_ids()

    def _validate_ids(self):
        if self.list_training_ids is None and self.predict_id is None:
            raise ValueError(
                "At least one of list_training_ids or predict_id must be provided."
            )

        if len(self.list_training_ids) > 1 and self.fold is None:
            raise ValueError(
                "Only one ID is supported for training when fold is not specified."
            )

    def prepare_single_adata(self, id: str) -> AnnData:
        logger.info(f"Preparing data for ID: {id}.")

        # Load and preprocess data
        embedding_path = (
            self.data_dir / "cell_embeddings" / f"{self.embedding_tag}_{id}.h5"
        )
        adata = self.dataset_handler.load_and_preprocess_data(
            self.data_dir,
            id,
            self.filter_genes,
            self.filter_cells,
            self.normalize,
            self.log1p,
            embedding_path,
        )

        # Convert sparse to dense matrix if needed
        if issparse(adata.X):
            adata.X = adata.X.toarray()

        # Select genes
        self.genes_to_pred = select_labels(adata, self.genes)
        adata.raw = adata.copy()
        adata = adata[:, self.genes_to_pred].copy()

        return adata

    def prepare_data(self):
        if self.predict_id is not None:
            logger.info(f"Preparing data for prediction.")
            self.adata = self.prepare_single_adata(self.predict_id)
        else:
            logger.info(f"Preparing data for training.")
            self.adata_dict = {
                id: self.prepare_single_adata(id) for id in self.list_training_ids
            }

            # Ensure all slides share the same genes by taking the sorted intersection
            if len(self.adata_dict) > 1:
                logger.info(
                    "Computing the sorted intersection of genes across all slides."
                )

                # Get the intersection of all gene sets
                gene_sets = [set(adata.var_names) for adata in self.adata_dict.values()]
                self.genes_to_pred = sorted(set.intersection(*gene_sets))

                if not self.genes_to_pred:
                    raise ValueError("No common genes found across the slides.")

                # Subset each AnnData object to the common genes
                for key, adata in self.adata_dict.items():
                    self.adata_dict[key] = adata[:, self.genes_to_pred].copy()

                # Check r for each anndata
                list_r = list(
                    set(
                        [
                            int(
                                adata.uns["spatial"]["ST"]["scalefactors"][
                                    "spot_diameter_fullres"
                                ]
                            )
                            // 2
                            for adata in self.adata_dict.values()
                        ]
                    )
                )
                assert (
                    len(list_r) == 1
                ), f"Found mutliple d values in training dataset: {list_r}"
                self.r = list_r[0]

                logger.info(
                    f"Selected {len(self.genes_to_pred)} common genes across all slides."
                )

        logger.info("Data preparation completed.")

    def setup(self, stage: str | None = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            # Multiple slide training
            if len(self.list_training_ids) > 1:
                if self.fold == -1:
                    logger.info(f"Using all training slides: {self.list_training_ids}.")
                else:
                    out_slide = self.list_training_ids.pop(self.fold)
                    logger.info(f"Removing {out_slide} from training set.")
                    del self.adata_dict[out_slide]
                    logger.info(
                        f"Using {list(self.adata_dict.keys())} as training set."
                    )
            self.adata = ad.concat(
                self.adata_dict.values(), label="batch", uns_merge="first"
            )

            # Apply scaling
            if self.scale == "global_scaling":
                logger.info("Applying robust scaling to all data.")
                scaler = RobustScaler(with_centering=False, quantile_range=(0, 95.0))
                scaler.fit(self.adata.X)
                for key, adata in self.adata_dict.items():
                    adata.X = scaler.transform(adata.X)
            elif self.scale == "slide_scaling":
                logger.info("Applying robust scaling to data slide-by-slide.")
                for key, adata in self.adata_dict.items():
                    scaler = RobustScaler(
                        with_centering=False, quantile_range=(0, 95.0)
                    )
                    scaler.fit(self.adata.X)
                    adata.X = scaler.fit_transform(adata.X)
            else:
                logger.info("No scaling applied")

            # Create concatenated adata
            self.adata = ad.concat(
                self.adata_dict.values(), label="batch", uns_merge="first"
            )

            # Data split
            all_datasets = [
                self.dataset_handler.create_dataset(adata, self.data_dir)
                for adata in self.adata_dict.values()
            ]
            full_dataset = ConcatDataset(all_datasets)
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [self.frac_train, 1 - self.frac_train], generator=self.gen
            )
            logger.info(
                f"Using {len(self.train_dataset)} datapoint for training and {len(self.val_dataset)} for validation."
            )

        # Assign Test split(s) for use in Dataloaders
        if stage == "predict":
            self.predict_dataset = self.dataset_handler.create_dataset(
                self.adata, self.data_dir
            )
            if type(self.dataset_handler) in [
                HisToGeneVisiumHandler,
                THItoGeneVisiumHandler,
            ]:
                logger.info("Setting dataset train flag to false for predictions.")
                self.predict_dataset.train = False

        # Assign Test split(s) for use in Dataloaders
        if stage == "inference":
            self.custom_collate = None
            self.predict_dataset = self.dataset_handler.create_inference_dataset(
                embedding_path=self.adata.uns["cell_embedding_path"]
            )

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate,
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_dataset, self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return self._create_dataloader(
            self.predict_dataset, self.batch_size, shuffle=False
        )

    def get_gene_names(self) -> list[str]:
        return self.genes_to_pred


def prepare_data_module(config: DictConfig, stage: str, task_type: str) -> STDataModule:
    # Get data handler
    handler_map = {
        "supervised": SupervisedInstanceHandler,
        "mil": MilVisiumHandler,
        "xenium": XeniumHandler,
        "mclstexp_visium": MclSTExpVisiumHandler,
        "histogene_visium": HisToGeneVisiumHandler,
        "thitogene_visium": THItoGeneVisiumHandler,
        "istar_visium": IstarGeneVisiumHandler,
    }
    dataset_handler_cls = handler_map.get(config.dataset_handler)

    # Update data.config in case of NB dist
    if task_type in [TaskType.nb_mean_regression, TaskType.nb_total_regression]:
        logger.info(f"Using {task_type}, setting log-norm-scaling to false.")
        config.normalize = False
        config.log1p = False
        config.scale = "no_scaling"

    data_module = STDataModule(
        data_dir=config.data_dir,
        list_training_ids=config.list_training_ids,
        predict_id=config.predict_id,
        genes=config.genes,
        embedding_tag=config.embedding_tag,
        dataset_handler=dataset_handler_cls(),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        filter_genes=config.filter_genes,
        filter_cells=config.filter_cells,
        normalize=config.normalize,
        log1p=config.log1p,
        scale=config.scale,
        frac_train=config.frac_train,
        fold=config.fold,
        seed=config.seed,
    )
    data_module.prepare_data()
    data_module.setup(stage=stage)
    return data_module
