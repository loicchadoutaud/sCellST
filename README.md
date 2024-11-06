# sCellST

sCellST is a novel method for inferring gene expression from H&E images trained on spatial transcriptomics data.

![figure](method.png)

## Using the repository

To install the package, we recommend using poetry. 
More information on how to install poetry can be found [here](https://python-poetry.org/docs/).
Clone this repository and simply run the following command  in the repository to install a python environment with all the dependencies:
It has been tested on a linux machine with cuda 11.8.

```bash
git clone https://github.com/loicchadoutaud/sCellST.git
cd sCellST
conda env create -f environment.yml
conda activate scellst
```

## Organization of the repository

This repository is organized as follows:

- `hovernet` contains the code necessary to segment cells on the H&E images. 
It is a fork of the HoverNet repository, which can be found [here](https://github.com/vqdang/hover_net).
- `moco-v3` contains the code necessary to train the MoCo-v3 model on the segmented cells.
It is based on the official MoCo-v3 repository, which can be found [here](https://github.com/facebookresearch/moco-v3).

For both repositories, the modifications performed for this project can be found at the top of the original README files.

- `data`: contains the genes used in the paper when not highly variable genes
- `data_embedder`: contains the code necessary to prepare data before training the GE predictor (sCellST)
- `scellst`: contains the code necessary to train the GE predictor (sCellST)
- `scripts`: contains scripts to run the different steps of the pipeline
- `simulation`: contains the code necessary to create simulated data 

## Data download 

To run the demo, we used a Visium slide from the 10x Genomics website: [link](https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0).
We provide a Makefile to download the data and create the necessary folders.

To download the data, simply run the following command:
This will create the dataset folder and download the data in the appropriate folders.
The data downloaded is:
- The raw WSI
- The Visium data
- The pretrained weights for HoverNet

The following structure will be created:
```bash
dataset
├── raw_wsi
    └── Visium_FFPE_Human_Breast_Cancer_image.tif
└── Visium
    └── Visium_FFPE_Human_Breast_Cancer
        ├── filtered_feature_bc_matrix.h5
        └── spatial
```

If you want to apply to your own data, make sure that the wsi and the associated visium dat starts with the same name.

```bash
make download_data
```

## Minimal example
We also provide in the Makefile the steps to run the full pipeline on the Visium slide downloaded above (around 2 hours to complete).
The script will perform in order:
1. Convert wsi to pyramidal file 
2. Segment cells
3. Extract cells
4. Train MoCo-v3
5. Preprocess Visium data for the sCellST model
6. Train sCellST
7. Predict gene expression on cells from the Visium slide

To run the demo, simply run the following command after downloading the data:

```bash
make run_pipeline
```

## Citation

If you use this code, please cite the following paper:
