import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from anndata import AnnData
from loguru import logger
from pandas import Series
from scipy.spatial import KDTree
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop, ToImage, ToDtype, Compose, Normalize
from tqdm.auto import tqdm

from scellst.dataset.cell_image_dataset import CellH5HESTDataset
from scellst.module.image_encoder import InstanceEmbedder


def find_spot_containing_cells_vectorized(
    adata: AnnData,
    coords_center: np.ndarray,
    spot_radius_px: float,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Find the spot containing each cell using chunked vectorized operations.

    Args:
        adata (anndata.AnnData): anndata object
        coords_center (np.array): coordinates of the cell centers
        spot_radius_px (float): radius of the spot in pixels
        chunk_size (int): Number of cells to process per chunk for memory efficiency.

    Returns:
        np.array: Spot containing each cell.
    """
    spatial_coords = adata.obsm["spatial"]
    tree = KDTree(spatial_coords)
    spot_containing_cells = np.full(len(coords_center), "None", dtype="object")

    # Process cells in chunks to manage memory usage
    for start_idx in tqdm(range(0, len(coords_center), chunk_size)):
        end_idx = min(start_idx + chunk_size, len(coords_center))
        coords_center_chunk = coords_center[start_idx:end_idx]

        # Compute the distance between each cell and each spot
        indices = tree.query_ball_point(
            coords_center_chunk, r=spot_radius_px, workers=-1
        )

        # Flatten the indices and format them into a 2D NumPy array
        formatted_indices = np.array(
            [[i, value] for i, sublist in enumerate(indices) for value in sublist]
        )

        if formatted_indices.size == 0:
            continue

        # Extract the spot IDs for cells within the radius
        global_indices = (
            start_idx + formatted_indices[:, 0]
        )  # Add start_idx to the cell indices
        spot_ids = formatted_indices[:, 1]  # Spot indices
        spot_containing_cells[global_indices] = adata.obs_names[spot_ids]

    logger.info(
        f"Found {np.sum(spot_containing_cells != 'None')} / {len(spot_containing_cells)} = {np.sum(spot_containing_cells != 'None') / len(spot_containing_cells) * 100 :.2f} % cells within {len(adata)} spots."
    )
    logger.info(
        f"Found {len(np.unique(spot_containing_cells))} / {len(adata)} spots with cells."
    )
    return spot_containing_cells


def create_spot_cell_map(
    cell_embedding_path: str,
) -> Series:
    """Create a mapping from spot to cells.

    Args:
        cell_embedding_path (str): path to the cell embedding file

    Returns:
        Series: mapping from spot to cells
    """
    with h5py.File(cell_embedding_path, "r") as f:
        df = pd.DataFrame(
            {
                "spot": f["spot"][:].astype(str).flatten(),
                "barcode": f["barcode"][:].flatten(),
            }
        )
        ser = df.groupby("spot")["barcode"].apply(
            lambda x: x.index.sort_values().tolist()
        )
        if "None" in ser:
            ser.drop("None", axis=0, inplace=True)
    return ser


def save_hdf5(
    output_fpath, asset_dict, attr_dict=None, mode="a", auto_chunk=True, chunk_size=None
):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f:  # if key does not exist, create dataset
                data_type = val.dtype
                if data_type.kind in {"S"} or data_type == np.object_:
                    data_type = h5py.string_dtype(encoding="utf-8")
                if auto_chunk:
                    chunks = True  # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(
                        key,
                        shape=data_shape,
                        chunks=chunks,
                        maxshape=(None,) + data_shape[1:],
                        dtype=data_type,
                    )
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")

            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = val

    return output_fpath


def load_eval_transform(stats_path: Path) -> Compose:
    with open(stats_path, "r") as f:
        norm_dict = json.load(f)
    mean = norm_dict["mean"]
    std = norm_dict["std"]
    logger.info("Loaded mean/std normalisation.")
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=mean, std=std),
            CenterCrop(size=(48, 48)),
        ]
    )


def post_collate_fn(batch):
    """
    Post collate function to clean up batch
    """
    if batch["img"].dim() == 5:
        assert batch["img"].size(0) == 1
        batch["img"] = batch["img"].squeeze(0)
    if batch["coords"].dim() == 3:
        assert batch["coords"].size(0) == 1
        batch["coords"] = batch["coords"].squeeze(0)
    if batch["label"].dim() == 2:
        assert batch["label"].size(0) == 1
        batch["label"] = batch["label"].squeeze(0)
    if "barcode" in batch.keys():
        batch["barcode"] = np.stack(batch["barcode"]).flatten()
    if "spot" in batch.keys():
        batch["spot"] = np.stack(batch["spot"]).flatten()
    return batch


def embed_cells(
    dataloader: DataLoader,
    model: torch.nn.Module,
    embedding_save_path: str,
    device: str,
    precision,
    attr_dict: dict,
):
    """Extract embeddings from tiles using `encoder` and save to a h5 file"""
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch["img"].to(device).float()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            embeddings = model(imgs)
        if batch_idx == 0:
            mode = "w"
        else:
            mode = "a"
        asset_dict = {"embedding": embeddings.cpu().numpy()}
        asset_dict.update(
            {key: np.array(val) for key, val in batch.items() if key != "img"}
        )
        save_hdf5(
            embedding_save_path, asset_dict=asset_dict, attr_dict=attr_dict, mode=mode
        )
    return embedding_save_path


@torch.inference_mode()
def predict_cell_dataset(
    dataset_path: str,
    stats_path: str,
    model_name: str,
    weights_path: str,
    save_path: str,
    device: str,
    batch_size: int = 2048,
    num_workers: int = 4,
) -> None:
    logger.info(f"Embedding cells using {model_name} encoder")
    encoder = InstanceEmbedder(model_name, weights_path).to(device)
    encoder.eval()
    encoder.to(device)

    # Load attribute dict
    with h5py.File(dataset_path, mode="r") as f:
        attr_dict = {"embedding": dict(f["img"].attrs)}
        logger.info(f"Adding attribute dict: {attr_dict}")

    cell_dataset = CellH5HESTDataset(
        dataset_path,
        chunk_size=batch_size,
        img_transform=load_eval_transform(stats_path),
    )
    cell_dataloader = torch.utils.data.DataLoader(
        cell_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    logger.info(f"Starting encoding process...")
    embed_cells(cell_dataloader, encoder, save_path, device, torch.float32, attr_dict)
    logger.info(f"Done.")


def compute_mean_std(
    dataset_path: str,
    batch_size: int = 2048,
    num_workers: int = 0,
) -> dict:
    logger.info(f"Computing mean/std.")
    transform = Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            CenterCrop(48),
        ]
    )
    cell_dataset = CellH5HESTDataset(
        dataset_path, chunk_size=batch_size, img_transform=transform
    )
    cell_dataloader = torch.utils.data.DataLoader(
        cell_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    logger.info("Calculating mean and std of the dataset...")
    mean = 0
    std = 0
    for batch in tqdm(cell_dataloader):
        # Rearrange batch to be the shape of [B, C, W * H]
        img = batch["img"].squeeze(0)
        img = img.view(img.size(0), img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)

    mean /= len(cell_dataloader) * batch_size
    std /= len(cell_dataloader) * batch_size

    return {"mean": mean.tolist(), "std": std.tolist()}
