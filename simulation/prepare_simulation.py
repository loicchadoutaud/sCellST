import argparse
import os

import h5py
import numpy as np
import anndata as ad
from anndata import AnnData
from tqdm.auto import tqdm


def subset_embedding_adata(adata: AnnData, cluster_col: str, map_dict: dict,n_cells: str = 2000) -> AnnData:
    cell_names = []
    adata = adata[adata.obs[cluster_col].isin(map_dict.keys())]
    for cluster in adata.obs[cluster_col].unique():
        # Find the closest cells to center
        idx_closest = np.argsort(adata.obsm[f"{cluster_col}_dist"], axis=0)[
            :n_cells, cluster
        ].flatten()
        cell_names.extend(adata.obs_names[idx_closest].tolist())
        print(len(cell_names))
    cell_names = list(set(cell_names))  # to remove potential duplicates
    return adata[np.asarray(cell_names)]


def get_mapping() -> dict:
    return {
        0: "fallopian tube secretory epithelial cell",
        11: "fibroblast",
        6: "monocyte",
        4: "plasma cell",
        2: "endothelial cell",
        13: "T cell",
    }


def add_gene_expression(
    emb_adata: AnnData,
    ref_adata: AnnData,
    map_dict: dict,
    cluster_col: str,
    mode: str,
    celltype_col: str = "cell_type",
) -> AnnData:
    X = np.zeros((emb_adata.shape[0], ref_adata.shape[1]), dtype=np.int32)
    clusters = emb_adata.obs[cluster_col].values
    select_dict = {
        c: ref_adata[ref_adata.obs[celltype_col] == map_dict[c]].obs_names for c in np.unique(clusters)
    }
    cluster_exp = {
        c: np.asarray(ref_adata[ref_adata.obs[celltype_col] == map_dict[c]].X.mean(axis=0)) for c in np.unique(clusters)
    }
    for i in tqdm(range(emb_adata.shape[0])):
        match mode:
            case "cell":
                name = np.random.choice(
                    a=select_dict[clusters[i]],
                    size=1,
                )
                X[i] = ref_adata[name].X.toarray().squeeze()
            case "centroid":
                X[i] = cluster_exp[clusters[i]].squeeze()
            case "random":
                name = np.random.choice(
                    a=ref_adata.obs_names,
                    size=1,
                )
                X[i] = ref_adata[name].X.toarray().squeeze()
            case _:
                raise ValueError("Should be one of ['cell', 'centroid', 'random']")

    return AnnData(X=X, obs=emb_adata.obs, obsm=emb_adata.obsm, var=ref_adata.var)


def split_train_test(adata: AnnData, seed: int = 0) -> tuple[AnnData, AnnData]:
    rng = np.random.default_rng(seed=seed)
    cell_indexes = rng.permutation(adata.obs_names.to_numpy())
    n_cells = adata.shape[0] // 2
    return adata[cell_indexes[:n_cells]], adata[cell_indexes[n_cells:]]


def prepare_spot_data(
    cell_adata: AnnData, n_spots: int, n_cells_per_spot: int, cluster_tag: str
) -> AnnData:
    # Prepare variables to fill
    cell_adata.obs_names = np.arange(len(cell_adata)).astype(str)
    X = np.zeros((n_spots, cell_adata.shape[1]), dtype=np.float32)
    spot_cell_map, spot_cell_distance = {}, {}

    # Sample cells for each spots
    cell_in_spots = np.random.choice(
        cell_adata.obs_names, size=n_spots * n_cells_per_spot
    )

    # Prepare spot information
    idx_start = 0
    for i in tqdm(range(n_spots)):
        spot_cell_map[str(i)] = cell_in_spots[idx_start : idx_start + n_cells_per_spot].astype(int)
        spot_cell_distance[str(i)] = n_cells_per_spot * [0]
        X[i] = (
            cell_adata[cell_in_spots[idx_start : idx_start + n_cells_per_spot]]
            .X.toarray()
            .sum(axis=0)
        )
        idx_start += n_cells_per_spot

    # Create final anndata
    adata = AnnData(
        X=X,
        var=cell_adata.var,
    )
    adata.uns["spatial"] = {"simulation": {}}
    adata.uns["MIL"] = {}
    adata.uns["MIL"][f"spot_cell_map"] = spot_cell_map
    adata.uns["MIL"][f"spot_cell_distance"] = spot_cell_distance
    adata.uns["MIL"]["cell_label"] = cell_adata.obs[cluster_tag].values
    adata.uns["MIL"]["cell_coordinates"] = np.random.randint(
        0, 100, size=(len(cell_adata.obs["class"].values), 2)
    )
    adata.uns["MIL"]["slide_name"] = "simulation"
    adata.uns["MIL"]["physical_radius"] = 100
    return adata


def save_dataset(
    spot_adata: AnnData,
    cell_adata: AnnData,
    output_folder_path: str,
    tag: str,
) -> None:
    os.makedirs(output_folder_path, exist_ok=True)
    # Mil dataset
    os.makedirs(os.path.join(output_folder_path, "mil"), exist_ok=True)
    spot_adata.write_h5ad(
        os.path.join(output_folder_path, "mil", "adata_with_mil.h5ad")
    )
    h5file = h5py.File(os.path.join(output_folder_path, 'mil', f'embedding_moco_{tag}.h5'), "w")
    h5file.create_dataset(name="embeddings", shape=cell_adata.obsm["X_ovarian"].shape, dtype=np.float32, data=cell_adata.obsm["X_ovarian"])
    h5file.close()

    # Cell labels
    cell_adata.write_h5ad(os.path.join(output_folder_path, "cell_adata_labels.h5ad"))


def prepare_datasets(
    embedding_path: str,
    reference_path: str,
    tag: str,
    n_spots: int,
    n_cells_per_spot: int,
    cluster_key: str,
    obsm_key: str,
    output_folder_path: str,
    mode: str,
    random_seed: int = 0,
) -> None:
    # Set random seed
    np.random.seed(random_seed)

    # Load data
    print("Loading embedding...")
    emb_adata = ad.read_h5ad(embedding_path)
    print("Loading reference...")
    ref_adata = ad.read_h5ad(reference_path).raw.to_adata()
    ref_adata.var["feature_name"] = ref_adata.var["feature_name"].astype(str)
    ref_adata.var.set_index("feature_name", inplace=True)
    ref_adata.var_names_make_unique()

    # Subset embeddings
    print("Subsetting anndata")
    map_dict = get_mapping()
    emb_adata = subset_embedding_adata(emb_adata, f"{cluster_key}_{obsm_key}", map_dict)
    print(f"embedding adata shape: {emb_adata.shape}")

    # Add gene expression per clusters
    print("Attributing gene expressions...")
    cell_adata = add_gene_expression(
        emb_adata, ref_adata, map_dict, f"{cluster_key}_{obsm_key}", mode,
    )
    print(f"cell adata shape: {cell_adata.shape}")

    # Split train/test
    print("Splitting train/test data...")
    cell_adata_train, cell_adata_test = split_train_test(cell_adata)

    # Create spot dataset
    print("Creating spot adata...")
    spot_adata_train = prepare_spot_data(cell_adata_train, n_spots, n_cells_per_spot, f"{cluster_key}_{obsm_key}")
    spot_adata_test = prepare_spot_data(cell_adata_test, n_spots, n_cells_per_spot, f"{cluster_key}_{obsm_key}")

    # Save dataset
    print("Saving dataset...")
    save_dataset(
        spot_adata_train,
        cell_adata_train,
        output_folder_path + "_train",
        tag,
    )
    save_dataset(
        spot_adata_test,
        cell_adata_test,
        output_folder_path + "_test",
        tag,
    )


if __name__ == "__main__":
    print("Starting main script...")
    parser = argparse.ArgumentParser(description="Create simulated dataset")
    parser.add_argument(
        "--embedding_path", type=str, help="Path to anndata with embeddings"
    )
    parser.add_argument(
        "--reference_path", type=str, help="Path to scRNA reference dataset"
    )
    parser.add_argument("--n_spots", type=int, help="number of spots (train + test)")
    parser.add_argument(
        "--n_cells_per_spot", type=int, help="number of cells per spots"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="ovarian_resnet50_12.0",
        help="Tag to mimic debulk preprocessing",
    )

    parser.add_argument("--cluster_key", type=str, default="kmeans", help="cluster_key")
    parser.add_argument(
        "--obsm_key", type=str, default="ovarian", help="key for embeddings to use"
    )
    parser.add_argument(
        "--output_folder_path", type=str, help="folder to store dataset."
    )

    args = parser.parse_args()
    for mode in ["cell", "centroid", "random"]:
        prepare_datasets(
            embedding_path=args.embedding_path,
            reference_path=args.reference_path,
            tag=args.tag,
            n_spots=args.n_spots,
            n_cells_per_spot=args.n_cells_per_spot,
            cluster_key=args.cluster_key,
            obsm_key=args.obsm_key,
            output_folder_path=os.path.join(args.output_folder_path, mode),
            mode=mode,
        )
    print("End of main script.")
