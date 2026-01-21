
# sCellST

sCellST is a novel method for inferring gene expression from H&E images trained on spatial transcriptomics data.

![figure](method.jpg)

# Installation
The code has been tested with python 3.10 and cuda 11.8. To install the envrionment we used poetry which you can install from [poetry](https://python-poetry.org/). Make sure to use a version at least >= 2.0.
Then, you can simply run your code with `poetry run`

```bash
git clone https://github.com/loicchadoutaud/sCellST.git
cd sCellST
poetry install
poetry add --editable ./external/cell_SSL   # For SSL training
poetry run python full_pipeline_script.py
```

# Code structure

The code now used the HEST database (https://github.com/mahmoodlab/HEST) as raw data since it allows to use many publicly released spatial transcriptomics.

The code is structured as follows:
- scellst: source code for the sCellST method + benchmarked methods in scellst/bench
- external: external code used in the project
- sCellST_reproducibility/reproducibility_figures contains all the notebooks used to produce the figures from the paper
- sCellST_reproducibility/submit_scripts scripts contains all the scripts used to submit the jobs on the cluster with submitit (https://github.com/facebookincubator/submitit)
- simulation: contains all the code for the simulation experiments

# Usage

We provide a notebook to run the full pipeline on a breast cancer slide from the HEST database in training_tutorial.ipynb (takes between 10 and 15 minutes).

# Download reference data

```bash
# Reference scRNA-seq dataset
# 
wget https://datasets.cellxgene.cziscience.com/73fbcec3-f602-4e13-a400-a76ff91c7488.h5ad -O data/raw_ovary_dataset.h5ad
https://www.nature.com/articles/s41588-021-00911-1
wget https://datasets.cellxgene.cziscience.com/fabd4946-3f41-459c-ba79-188749a8baa4.h5ad -O data/raw_breast_dataset.h5ad
```

## Full text link
https://www.nature.com/articles/s41467-025-67965-1

```bash
@article{chadoutaud_scellst_2026,
	title = {{sCellST} predicts single-cell gene expression from {H}\& {E} images},
	url = {https://www.nature.com/articles/s41467-025-67965-1},
	doi = {10.1038/s41467-025-67965-1},
	journal = {Nature Communications},
	author = {Chadoutaud, Loïc and Lerousseau, Marvin and Herrero-Saboya, Daniel and Ostermaier, Julian and Fontugne, Jacqueline and Barillot, Emmanuel and Walter, Thomas},
	year = {2026},
}

```

## Credits
We thanks the authors of the HEST database (https://github.com/mahmoodlab/HEST) and the original Mocov3 (https://github.com/facebookresearch/moco-v3) adapted for this project 
