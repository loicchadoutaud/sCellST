
# sCellST

sCellST is a novel method for inferring single cell gene expression from H&E images trained on spatial transcriptomics data.

![figure](method.jpg)

# Installation
The code has been tested with python 3.10 and cuda 11.8

```bash
git clone git@github.com:loicchadoutaud/sCellST.git
cd sCellST
conda env create -f environment.yml
conda activate sCellST
pip install git+https://github.com/mahmoodlab/HEST.git
CODEPATH=$(realpath .)
conda env config vars set PYTHONPATH=$PYTHONPATH:$CODEPATH/external/cell_SSL
conda deactivate
conda activate sCellST
```

# Code structure

The code now used the HEST database (https://github.com/mahmoodlab/HEST) as raw data since it allows to use many publicly released spatial transcriptomic dataset.

The code is structured as follows:
- **config**: configuration files for the training and the inference.
- **data**: list of genes used for the training.
- **external**: external code used in the project
- **scellst**: source code for the sCellST method + script to launch benchmarked methods in scellst/bench.
- **simulation**: contains all the code for the simulation experiments

# Usage

We provide a simple python script to explain the main step of the pipeline: 

```python
from pathlib import Path
from scellst.submit_function import download_data, run_ssl, embed_cells
from scellst.train import train_and_save
from scellst.predict import predict_and_save

### Download data
path_dataset = Path("hest_data")
list_sample_ids = ["TENX39"]
download_data(path_dataset, None, list_sample_ids)

### Embed cells
# # SSL training
# tag = "moco-TENX39-rn50"
# run_ssl(path_dataset, None, SAMPLE_IDS, tag, 2, 4)

# TL
tag = "imagenet-rn50"

embed_cells(path_dataset, None, list_sample_ids, tag, "resnet50", "train")

### Train model
additional_kwargs = {
    "data_dir": path_dataset,
    "save_dir_tag": "test",
    "embedding_tag": f"{tag}_train",
    "genes": "HVG:1000",
    "list_training_ids": list_sample_ids,
}
config_path = Path("config/gene_default.yaml")
train_and_save(config_path, additional_kwargs)

### Predict
exp_tag = "embedding_tag=imagenet-rn50_train;genes=HVG:1000;train_slide=TENX39"
config_dir = Path("models") / "mil" / "test" / "exp_tag"
additional_kwargs = {"predict_id": list_sample_ids[0]}
infer_mode = "bag"  # or instance to have cell level outputs
predict_and_save(config_dir, additional_kwargs, infer_mode, save_adata=True)
```
(runtime around 10 minutes without GPU.)

1) Download the data: choose the ids in the hest database you want to use.
2) Embed the cells: choose the model you want to use to embed the cells. (optional but recommended for best results: train a SSL model on the data)
3) Train the model: train the model on the data.
4) Predict: predict the gene expression of the cells.

(If you encounter an issue related to the number of open files, consider increasing the limit with `ulimit -n 1048576`)


# Download reference data (only useful for simulations)

```bash
# Reference scRNA-seq dataset for simulation
wget https://datasets.cellxgene.cziscience.com/73fbcec3-f602-4e13-a400-a76ff91c7488.h5ad -O data/raw_ovarian_dataset.h5ad
```

## Preprint

If you look for the preprint, please refer to this link:

https://www.biorxiv.org/content/early/2024/11/08/2024.11.07.622225

```bash

@article{chadoutaud_scellst_2024,
	title = {{sCellST}: a {Multiple} {Instance} {Learning} approach to predict single-cell gene expression from {H}\&{E} images using spatial transcriptomics},
	doi = {10.1101/2024.11.07.622225},
	journal = {bioRxiv},
	author = {Chadoutaud, Loic and Lerousseau, Marvin and Herrero-Saboya, Daniel and Ostermaier, Julian and Fontugne, Jacqueline and Barillot, Emmanuel and Walter, Thomas},
	year = {2024}
}
```

If you look for the scripts used to generate the figures in the paper, please refer to this [repo](https://github.com/loicchadoutaud/sCellST_reproducibility).