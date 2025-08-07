import pandas as pd
from functools import reduce
from pandas import DataFrame


def format_metric_df(df: DataFrame, metric_list: list[str]) -> DataFrame:
    # Melt all columns that match any metric
    metrics_prefixes = tuple(f"{metric}/" for metric in metric_list)
    melted_df = df.melt(
        id_vars=["tag"],
        value_vars=[col for col in df.columns if col.startswith(metrics_prefixes)],
        var_name="metric_gene",
        value_name="value",
    )

    # Extract 'metric' and 'gene' from the column names like 'metric/gene'
    melted_df[["metric", "gene"]] = melted_df["metric_gene"].str.split("/", expand=True)

    # Drop the original combined column
    melted_df = melted_df.drop(columns="metric_gene")

    # Pivot so that each metric is a separate column again
    df_merged = melted_df.pivot_table(
        index=["tag", "gene"], columns="metric", values="value"
    ).reset_index()

    # Optional: Drop rows with missing values
    if df_merged.isna().any().any():
        print(f"Be careful, {df_merged.isna().sum().sum()} NaN found in the metrics\n{df_merged.isna().sum()}")
    return df_merged
