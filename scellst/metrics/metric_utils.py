import pandas as pd
from pandas import DataFrame


def format_metric_df(df: DataFrame) -> DataFrame:
    """Format metric DataFrame for plotting from wide to long format."""
    pcc_cols = [col for col in df.columns if col.startswith("pcc/")]
    scc_cols = [col for col in df.columns if col.startswith("scc/")]

    df_pcc = df.melt(
        id_vars=["tag"], value_vars=pcc_cols, var_name="gene", value_name="pcc"
    )
    df_scc = df.melt(
        id_vars=["tag"], value_vars=scc_cols, var_name="gene", value_name="scc"
    )

    df_pcc["gene"] = df_pcc["gene"].str.split("/").str[1]
    df_scc["gene"] = df_scc["gene"].str.split("/").str[1]

    df = pd.merge(df_pcc, df_scc, on=["gene", "tag"])
    return df
