import pandas as pd
import scanpy as sc


if __name__ == '__main__':
    dataset_name = "ovarian"

    # Select only left/right ovary cells from raw dataset
    adata = sc.read_h5ad(f"../data/raw_{dataset_name}_dataset.h5ad", backed="r")
    adata = adata[adata.obs["tissue"].isin(["left ovary", 'right ovary'])].to_memory()
    adata.obs.drop(columns=["cell_id"], inplace=True)
    adata.write_h5ad(f"../data/{dataset_name}_dataset.h5ad")

    # Read data
    adata = sc.read_h5ad(f"../data/{dataset_name}_dataset.h5ad")
    adata = adata.raw.to_adata()
    adata.var_names = adata.var["feature_name"].astype(str)

    # Preprocess data
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_counts=10)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.highly_variable_genes(
        adata, n_top_genes=1000, subset=False, layer="counts", flavor="seurat_v3"
    )
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Extract marker genes
    sc.tl.rank_genes_groups(adata, groupby="cell_type", method="t-test")

    # Save dataframe
    df = sc.get.rank_genes_groups_df(adata, group=None)
    df["group"] = df["group"].replace({"B cell": "lymphocyte", "T cell": "lymphocyte"})
    top_scores = df.groupby("group")["scores"].nlargest(20)
    top_scores_with_string = pd.merge(
        top_scores,
        df[["group", "scores", "names"]],
        how="left",
        left_on=["group", "scores"],
        right_on=["group", "scores"],
    )

    # Keep only some of the annotations
    LIST_SCORE_TO_PLOT = [
        "fibroblast",
        "endothelial cell",
        "lymphocyte",
        "plasma cell",
        "fallopian tube secretory epithelial cell",
    ]
    top_scores_with_string = top_scores_with_string[top_scores_with_string["group"].isin(LIST_SCORE_TO_PLOT)]
    top_scores_with_string["group"] = top_scores_with_string["group"].cat.remove_unused_categories()
    top_scores_with_string.to_csv(f"../data/{dataset_name}_celltype_markers.csv")

    # Print to latex
    top_scores_with_string = top_scores_with_string.sort_values(by="names")
    top_scores_with_string.rename(columns={"group": "cell type", "names": "top 20 genes"}, inplace=True)
    df = top_scores_with_string.groupby("cell type")["top 20 genes"].apply(list)
    with pd.option_context("max_colwidth", 1000):
        print(df.to_latex(column_format="p{3cm}p{10cm}", escape=False))

