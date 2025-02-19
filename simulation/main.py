from pathlib import Path

from simulation.data_loader import SimulationDataLoader
from simulation.data_preparation import DataPreparation


def prepare_simulation(
    data_path: Path, ref_adata_path: Path, embedding_key: str, simulation_mode: str
) -> None:
    data_loader = SimulationDataLoader(
        data_path=data_path,
        embedding_key=embedding_key,
        ref_adata_path=ref_adata_path,
    )
    emb_adata, ref_adata = data_loader.load_data(
        subset_col="tissue", value_to_keep=["left ovary", "right ovary"]
    )
    data_prep = DataPreparation(
        emb_adata=emb_adata,
        ref_adata=ref_adata,
        simulation_mode=simulation_mode,
        data_path=data_path,
    )
    data_prep.prepare_simulations()
