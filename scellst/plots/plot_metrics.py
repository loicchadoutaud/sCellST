import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame


def plot_boxplots(
    df: DataFrame, metrics: list[str], params: list[str], save_path: str | None = None
) -> None:
    fig, axs = plt.subplots(
        len(metrics), len(params), figsize=(5 * len(params), 5 * len(metrics))
    )

    if len(metrics) == 1:
        axs = axs[np.newaxis, :]
    if len(params) == 1:
        axs = axs[:, np.newaxis]

    for i, metric in enumerate(metrics):
        for j, param in enumerate(params):
            sns.boxplot(x=param, y=metric, data=df, ax=axs[i, j])
            sns.stripplot(x=param, y=metric, data=df, color="black", ax=axs[i, j])
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
