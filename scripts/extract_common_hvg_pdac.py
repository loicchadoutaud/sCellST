import argparse
import os
from typing import List

import anndata as ad
import scanpy as sc
import pickle

from anndata import AnnData


def load_visium(data_folder: str) -> AnnData:
    adata = sc.read_visium(data_folder)
    adata.var_names_make_unique()
    return adata


def preprocess_adata(adata: AnnData, n_genes: int = 2000) -> AnnData:
    sc.pp.filter_genes(adata, min_counts=200)
    sc.pp.filter_genes(adata, min_cells=adata.shape[0] // 10)
    sc.pp.filter_cells(adata, min_counts=20)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata = adata[:, ~(adata.var["mt"])].copy()

    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_genes,
        layer="counts",
        flavor="seurat_v3",
        subset=False,
    )
    return adata


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_folder", type=str, required=True)
    args.add_argument("--output_folder", type=str, required=True)
    args.add_argument("--list_files", nargs="+", type=str, required=True)
    args = args.parse_args()

    list_adata_path = [os.path.join(args.data_folder, f) for f in args.list_files]
    list_adata = [preprocess_adata(load_visium(path)) for path in list_adata_path]
    list_adata_filtered = []
    for i in range(len(list_adata)):
        list_adata[i].obs["slide"] = os.path.basename(list_adata_path[i])
        list_adata_filtered.append(list_adata[i][:, list_adata[i].var.highly_variable])
    adata = ad.concat(list_adata_filtered)
    adata.obs_names_make_unique()
    print(len(adata.var_names))
    with open(os.path.join(args.output_folder, "gene_pdac.pkl"), "wb") as f:
        pickle.dump(adata.var_names.sort_values(), f)
    with open(os.path.join(args.output_folder, "gene-names.txt"), "w") as file:
        for s in adata.var_names.sort_values():
            file.write(f"{s}\n")
