from concurrent.futures import ThreadPoolExecutor
from loguru import logger

import os
import h5py
import numpy as np
from tqdm.auto import tqdm


def read_key(
    embeddings, n_threads: int = 4
) -> np.ndarray:
    # Optimal chunking strategy
    chunk_size = max(
        4096, len(embeddings) // (n_threads * 4)
    )  # At least 4K samples per chunk
    chunks = [
        (i, min(i + chunk_size, len(embeddings)))
        for i in range(0, len(embeddings), chunk_size)
    ]

    def read_chunk(start, end):
        return embeddings[start:end]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(read_chunk, start, end) for start, end in chunks]
        results = [f.result() for f in futures]
    return np.concatenate(results)


def subset_h5files(path: str, ids: list[str]) -> None:
    # Step 1: Read all cell_ids
    with h5py.File(path, "r") as f:
        all_ids = f["barcode"][:].astype(str).astype("object").flatten()
        id_to_index = {id_: idx for idx, id_ in enumerate(all_ids)}

        # Step 2: Find indices of desired cell_ids
        selected_indices = np.array([id_to_index[id_] for id_ in ids])

        # Step 3: Subset all datasets by those indices
        subset_data = {}
        for key in tqdm(f.keys(), total=len(f.keys()), desc="subset h5file"):
            logger.info(key)
            logger.info(f[key].shape)
            full_dataset = read_key(f[key])
            subset_data[key] = full_dataset[selected_indices]
            logger.info(subset_data[key].shape)

            # Additional check for barcodes
            if key == "barcode":
                print(ids[:5])
                print(subset_data[key][:5])

    # Step 4 (Optional): Save to new file
    dir_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    target_path = os.path.join(dir_path, "common_" + file_name)
    with h5py.File(target_path, "w") as f:
        for key, data in tqdm(subset_data.items(), total=len(subset_data.keys()), desc="write h5file"):
            f.create_dataset(key, data=data)
