from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import scanpy as sc
from anndata import AnnData
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

from scellst.constant import FIGURES_DIR, DATA_DIR
from scellst.plots.plot_ssl import plot_ssl_cell_gallery
from simulation.decoupler_utils import get_pseudobulk


class DataPreparation:
    def __init__(
        self,
        emb_adata: AnnData,
        ref_adata: AnnData,
        simulation_mode: str,
        data_path: Path,
        n_spots: int = 5000,
        n_cells_per_spot: int = 20,
        n_clusters: int = 6,
        n_cells_per_cluster: int = 2000,
        n_genes: int = 1000,
        random_seed: int = 0,
        cluster_col: str = "kmeans",
        celltype_col: str = "cell_type",
    ):
        self.emb_adata = emb_adata
        self.ref_adata = ref_adata
        self.simulation_mode = simulation_mode
        self.data_path = data_path
        self.n_spots = n_spots
        self.n_cells_per_spot = n_cells_per_spot
        self.n_clusters = n_clusters
        self.n_cells_per_cluster = n_cells_per_cluster
        self.n_genes = n_genes
        self.random_seed = random_seed
        self.cluster_col = cluster_col
        self.celltype_col = celltype_col
        self.int_celltype_col = f"int_{self.celltype_col}"
        self.rng = np.random.default_rng(seed=self.random_seed)

        assert n_clusters <= len(
            ref_adata.obs["cell_type"].unique()
        ), "Number of clusters should be less than the number of cell types in reference dataset"

    def cluster_embedding(self) -> None:
        X = PCA(n_components=50, random_state=self.random_seed).fit_transform(
            self.emb_adata.X
        )
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed)
        self.emb_adata.obs[self.cluster_col] = kmeans.fit_predict(X)
        self.emb_adata.obsm[f"{self.cluster_col}_dist"] = kmeans.transform(X)
        logger.info(
            f"Clustering done on embedding data with {self.n_clusters} clusters"
        )
        logger.info(
            f"Clusters: {np.unique(self.emb_adata.obs[self.cluster_col], return_counts=True)}"
        )

    def subset_embedding_adata(self) -> None:
        # Calculate distances and prepare the DataFrame
        cluster_dist = self.emb_adata.obsm[f"{self.cluster_col}_dist"]
        closest_cluster_dist = np.min(cluster_dist, axis=1)
        df_cluster_space = pd.DataFrame(
            {
                "distance": closest_cluster_dist,
                "cluster": self.emb_adata.obs[self.cluster_col].values,
            },
            index=self.emb_adata.obs_names,
        )

        # Select top cells for each cluster
        selected_indices = (
            df_cluster_space.groupby("cluster")["distance"]
            .nsmallest(self.n_cells_per_cluster)
            .index.get_level_values(1)
        )

        # Subset the AnnData object
        self.emb_adata = self.emb_adata[selected_indices].copy()

        logger.info(
            f"Selected {self.n_cells_per_cluster} cells per cluster from embedding data\n"
            f"Number of cells in embedding dataset: {self.emb_adata.shape[0]}"
        )

    def plot_selected_cluster(self, n_cell_per_cluster: int = 100) -> None:
        FIGURES_DIR.mkdir(exist_ok=True)
        save_fig_path = FIGURES_DIR / "sim"
        save_fig_path.mkdir(exist_ok=True)
        plot_ssl_cell_gallery(
            self.emb_adata,
            self.data_path,
            self.simulation_mode,
            save_fig_path,
            n_cell_per_cluster,
            "kmeans",
        )

    def preprocess_reference_adata(self) -> None:
        # Filtering
        sc.pp.filter_cells(self.ref_adata, min_genes=200)
        sc.pp.filter_genes(self.ref_adata, min_counts=10)

        # Normalisation
        self.ref_adata.layers["counts"] = self.ref_adata.X.copy()
        sc.pp.normalize_total(self.ref_adata)
        sc.pp.log1p(self.ref_adata)

        # HVG selection
        sc.pp.highly_variable_genes(
            self.ref_adata, n_top_genes=self.n_genes, layer="counts", flavor="seurat_v3"
        )
        logger.info("Preprocessed reference dataset.")

    def extract_and_save_all_marker_genes(self) -> None:
        # Extract marker genes
        sc.tl.rank_genes_groups(
            self.ref_adata, groupby=self.celltype_col, method="t-test_overestim_var"
        )
        df = sc.get.rank_genes_groups_df(self.ref_adata, group=None)
        print(df)
        df["group"] = df["group"].replace(
            {"B cell": "lymphocyte", "T cell": "lymphocyte"}
        )
        top_scores = df.groupby("group")["scores"].nlargest(20)
        top_scores_with_string = pd.merge(
            top_scores,
            df[["group", "scores", "names"]],
            how="left",
            left_on=["group", "scores"],
            right_on=["group", "scores"],
        )
        top_scores_with_string.rename(columns={"names": "gene"}, inplace=True)
        top_scores_with_string.to_csv(DATA_DIR / "genes_marker_ovary.csv")

    def select_cluster_ref_adata(self) -> None:
        self.ref_adata = self.ref_adata[:, self.ref_adata.var["highly_variable"]]

        celltype_adata = get_pseudobulk(
            adata=self.ref_adata,
            sample_col=self.celltype_col,
            groups_col=None,
            mode="mean",
        )
        kmedoids = KMedoids(
            n_clusters=self.n_clusters, random_state=self.random_seed
        ).fit(celltype_adata.X)
        idx_centroids = kmedoids.medoid_indices_
        cluster_to_keep = celltype_adata.obs_names[idx_centroids].sort_values().tolist()
        self.ref_adata = self.ref_adata[
            self.ref_adata.obs[self.celltype_col].isin(cluster_to_keep)
        ].copy()
        logger.info(
            f"Selected {self.n_clusters} clusters from reference dataset: {cluster_to_keep}\n"
            f"Number of cells in reference dataset: {self.ref_adata.shape[0]}"
        )

    def create_paired_cell_adata(self) -> None:
        # Convert cell type to integer
        self.ref_adata.obs[self.int_celltype_col] = self.ref_adata.obs[
            self.celltype_col
        ].factorize(sort=True)[0]

        # Create gene expression dataframe with clusters
        df_exp = pd.DataFrame(
            self.ref_adata.X.toarray(),
            index=self.ref_adata.obs_names,
            columns=self.ref_adata.var_names,
        )
        df_exp[self.int_celltype_col] = self.ref_adata.obs[
            self.int_celltype_col
        ]  # Add celltype clusters

        # Get embedding clusters
        embedding_clusters = self.emb_adata.obs[self.cluster_col].values

        match self.simulation_mode:
            case "centroid":
                cluster_exp_df = df_exp.groupby(self.int_celltype_col, sort=True).mean()
                X = cluster_exp_df.iloc[embedding_clusters][
                    self.ref_adata.var_names
                ].values
            case "cell":
                result = df_exp.groupby(self.int_celltype_col, sort=True).apply(
                    lambda x: self.rng.choice(x.index, size=self.emb_adata.shape[0]),
                    include_groups=False,
                )
                selected_names = np.stack(result.values).T[
                    np.arange(self.emb_adata.shape[0]), embedding_clusters
                ]
                X = df_exp.loc[selected_names, self.ref_adata.var_names].values
            case "random":
                selected_names = self.rng.choice(
                    self.ref_adata.obs_names.values, size=self.emb_adata.shape[0]
                )
                X = df_exp.loc[selected_names, self.ref_adata.var_names].values
            case _:
                raise ValueError("Should be one of ['cell', 'centroid', 'random']")

        self.paired_adata = AnnData(
            X=X,
            obs=self.emb_adata.obs,
            obsm={
                "embedding": self.emb_adata.X,
                "one_hot_encoded_celltype": pd.get_dummies(
                    self.emb_adata.obs[self.cluster_col]
                ).values.astype(int),
            },
            var=self.ref_adata.var,
        )
        logger.info("Added gene expression to embedding data.")

    def split_train_test(self):
        cell_indexes = self.rng.permutation(self.paired_adata.obs_names.to_numpy())
        n_cells = self.paired_adata.shape[0] // 2
        return (
            self.paired_adata[cell_indexes[:n_cells]],
            self.paired_adata[cell_indexes[n_cells:]],
        )

    def prepare_spot_data(self, cell_adata: AnnData) -> tuple[AnnData, AnnData]:
        # Select cells for each spot (there will be duplicated cells)
        cell_names_in_spot = self.rng.choice(
            cell_adata.obs_names, size=self.n_spots * self.n_cells_per_spot
        )

        # Create spot expression data
        cell_X = pd.DataFrame(
            cell_adata[cell_names_in_spot].X, columns=cell_adata.var_names
        )
        cell_X["spot_names"] = np.repeat(
            np.arange(self.n_spots), self.n_cells_per_spot
        ).astype(str)
        spot_X = cell_X.groupby("spot_names").sum()
        spot_adata = AnnData(
            X=spot_X.values,
            obs=pd.DataFrame(index=spot_X.index),
            var=pd.DataFrame(index=spot_X.columns),
        )

        # Create cell adata
        cell_adata = cell_adata[cell_names_in_spot].copy()
        cell_adata.obs["spot_names"] = cell_X["spot_names"].values

        logger.info("Prepared spot and cell data.")
        return spot_adata, cell_adata

    def save_dataset_to_hestdata(
        self, spot_adata: AnnData, cell_adata: AnnData, tag: str
    ) -> None:
        id_tag = f"sim_{self.simulation_mode}_{tag}"

        # Save st data
        spot_adata.write_h5ad(
            self.data_path / "st" / f"{self.emb_adata.uns['sample_id']}_{id_tag}.h5ad"
        )

        # Save cell_embedding
        save_h5file = (
            self.data_path
            / "cell_embeddings"
            / f"{self.emb_adata.uns['embedding_key']}_{self.emb_adata.uns['sample_id']}_{id_tag}.h5"
        )

        with h5py.File(save_h5file, "w") as h5_file:
            h5_file.create_dataset(
                "barcode", data=cell_adata.obs_names.values.astype(int)
            )
            h5_file.create_dataset("embedding", data=cell_adata.obsm["embedding"])
            h5_file.create_dataset("expression", data=cell_adata.X)
            h5_file.create_dataset(
                "label", data=cell_adata.obs[self.cluster_col].values[:, np.newaxis]
            )
            h5_file.create_dataset("spot", data=cell_adata.obs["spot_names"].values)
            h5_file.create_dataset("coords", data=np.zeros((len(cell_adata), 2)))
            h5_file.create_dataset(
                "gene_names",
                data=cell_adata.var_names.values,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        # Save one_hot_encoded embeddings
        save_h5file = (
            self.data_path
            / "cell_embeddings"
            / f"one-hot-celltype_{self.emb_adata.uns['sample_id']}_{id_tag}.h5"
        )

        with h5py.File(save_h5file, "w") as h5_file:
            h5_file.create_dataset(
                "barcode", data=cell_adata.obs_names.values.astype(int)
            )
            h5_file.create_dataset(
                "embedding", data=cell_adata.obsm["one_hot_encoded_celltype"]
            )
            h5_file.create_dataset("expression", data=cell_adata.X)
            h5_file.create_dataset(
                "label", data=cell_adata.obs[self.cluster_col].values[:, np.newaxis]
            )
            h5_file.create_dataset("spot", data=cell_adata.obs["spot_names"].values)
            h5_file.create_dataset("coords", data=np.zeros((len(cell_adata), 2)))
            h5_file.create_dataset(
                "gene_names",
                data=cell_adata.var_names.values,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

        logger.info(f"Saved dataset to hest_data with tag {tag}")

    def save_genes_to_pred(self) -> None:
        # Save hvg
        pd.Series(self.ref_adata.var_names, name="gene").sort_values().to_csv(
            DATA_DIR / f"genes_{self.n_genes}_hvg_sim.csv"
        )

        # Save marker genes
        df = sc.get.rank_genes_groups_df(
            self.ref_adata, group=self.ref_adata.obs[self.celltype_col].unique()
        )
        top_scores = df.groupby("group")["scores"].nlargest(20)
        top_scores_with_string = pd.merge(
            top_scores,
            df[["group", "scores", "names"]],
            how="left",
            left_on=["group", "scores"],
            right_on=["group", "scores"],
        )
        top_scores_with_string.sort_values(by="names", inplace=True)
        genes = top_scores_with_string["names"].drop_duplicates()
        pd.Series(genes.tolist(), name="gene").to_csv(
            DATA_DIR / f"genes_marker_sim.csv"
        )

    def prepare_simulations(self) -> None:
        # Prepare embedding data
        self.cluster_embedding()
        self.subset_embedding_adata()
        self.plot_selected_cluster()

        # Prepare ref data
        self.preprocess_reference_adata()
        self.extract_and_save_all_marker_genes()
        self.select_cluster_ref_adata()

        # Prepare paired data
        self.create_paired_cell_adata()
        self.save_genes_to_pred()

        # Prepare split
        train_adata, test_adata = self.split_train_test()

        # Format and save
        spot_train_adata, cell_train_adata = self.prepare_spot_data(train_adata)
        self.save_dataset_to_hestdata(spot_train_adata, cell_train_adata, "train")

        spot_test_adata, cell_test_adata = self.prepare_spot_data(test_adata)
        self.save_dataset_to_hestdata(spot_test_adata, cell_test_adata, "test")
