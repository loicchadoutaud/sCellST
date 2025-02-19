
# sCellST

sCellST is a novel method for inferring gene expression from H&E images trained on spatial transcriptomics data.

![figure](method.jpg)

# Installation
The code has been tested with python 3.10 and cuda 11.8

```bash
git clone --recurse-submodules git@github.com:loicchadoutaud/sCellST.git
cd sCellST
conda env create -f environment.yml
conda activate sCellST
CODEPATH=$(realpath .)
conda env config vars set PYTHONPATH=$PYTHONPATH:$CODEPATH:$CODEPATH/external/HEST/src:$CODEPATH/external/cell_SSL
conda deactivate
conda activate sCellST
```

# Code structure

The code now used the HEST database (https://github.com/mahmoodlab/HEST) as raw data since it allows to use many publicly released spatial transcriptomics.

The code is structured as follows:
- scellst: source code for the sCellST method + benchmarked methods in scellst/bench
- external: external code used in the project (as git submodules or code files when modification were necessary)
- reproducibility notebooks contains all the notebooks used to produce the figures from the paper
- submit scripts contains all the scripts used to submit the jobs on the cluster with submitit (https://github.com/facebookincubator/submitit)
- simulation: contains all the code for the simulation experiments

# Usage

We provide a notebook to run the full pipeline on a breast cancer slide from the HEST database in training_tutorial.ipynb.

# Download reference data

```bash
# Reference scRNA-seq dataset
wget https://datasets.cellxgene.cziscience.com/73fbcec3-f602-4e13-a400-a76ff91c7488.h5ad -O data/raw_ovarian_dataset.h5ad
```

## Preprint
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
