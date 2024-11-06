# gloabl variables
export PYTHONPATH := .:$(PYTHONPATH)
RANDOM := $(shell bash -c 'echo $$RANDOM')

# Variables
SLIDE := Visium_FFPE_Human_Breast_Cancer
TAG := demo
DATASET_PATH := dataset
WSI_PATH := $(DATASET_PATH)/wsi
WSI_OUT_PATH := $(DATASET_PATH)/wsi_out
PANNUKE_WEIGHTS := pretrained_weights/hovernet_fast_pannuke_type_tf2pytorch.tar
MOCO_WEIGHTS := moco-v3/best_model/moco_$(TAG)_model_best.pth.tar

# Default target
download_data: create_folders download_raw_wsi download_visium_data download_spatial_data download_hovernet_weights
run_pipeline: convert_to_pyramidal run_hovernet cell_extraction moco_training slide_preprocessing sCellST_training sCellST_inference
all: download_data run_pipeline

# Create dataset folders
create_folders:
	mkdir -p dataset/raw_wsi dataset/Visium/Visium_FFPE_Human_Breast_Cancer

# Download raw WSI image
download_raw_wsi: create_folders
	if [ ! -f dataset/raw_wsi/Visium_FFPE_Human_Breast_Cancer_image.tif ]; then \
		echo "Downloading raw WSI image..."; \
		wget -P dataset/raw_wsi https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_image.tif; \
	else \
		echo "Raw WSI image already downloaded."; \
	fi

# Download Visium filtered feature matrix
download_visium_data: create_folders
	if [ ! -f dataset/Visium/Visium_FFPE_Human_Breast_Cancer/filtered_feature_bc_matrix.h5 ]; then \
		echo "Downloading Visium filtered feature matrix..."; \
		wget -O dataset/Visium/Visium_FFPE_Human_Breast_Cancer/filtered_feature_bc_matrix.h5 https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5; \
	else \
		echo "Filtered feature matrix already downloaded."; \
	fi

# Download and extract Visium spatial data
download_spatial_data: create_folders
	if [ ! -d dataset/Visium/Visium_FFPE_Human_Breast_Cancer/spatial ]; then \
		echo "Downloading and extracting spatial data..."; \
		wget -P dataset/Visium/Visium_FFPE_Human_Breast_Cancer https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz; \
		tar -xvf dataset/Visium/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz -C dataset/Visium/Visium_FFPE_Human_Breast_Cancer; \
		rm dataset/Visium/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_spatial.tar.gz; \
	else \
		echo "Spatial data already downloaded and extracted."; \
	fi

# Download HoverNet pretrained weights
download_hovernet_weights:
	mkdir -p hovernet/pretrained_weights
	if [ ! -f hovernet/pretrained_weights/hovernet_fast_pannuke_type_tf2pytorch.tar ]; then \
		echo "Downloading HoverNet pretrained weights..."; \
		gdown https://drive.google.com/uc?id=1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR -O hovernet/pretrained_weights/hovernet_fast_pannuke_type_tf2pytorch.tar; \
	else \
		echo "HoverNet pretrained weights already downloaded."; \
	fi

# Convert raw WSI to pyramidal format
convert_to_pyramidal:
	python -u scripts/convert_to_pyr.py \
		--input_folder_path $(DATASET_PATH)/raw_wsi \
		--adata_folder_path $(DATASET_PATH)/Visium

# Run HoverNet inference on WSI
run_hovernet:
	cd hovernet && \
	python -u run_infer.py \
		--gpu='0' \
		--nr_types=6 \
		--type_info_path=type_info.json \
		--batch_size=64 \
		--model_mode=fast \
		--model_path=$(PANNUKE_WEIGHTS) \
		--nr_inference_workers=8 \
		--nr_post_proc_workers=8 \
		wsi \
		--input_dir=../$(WSI_PATH) \
		--output_dir=../$(WSI_OUT_PATH) \
		--cache_path=/tmp/hovernet_cache_$(RANDOM)\
		--save_thumb \
		--save_mask

# Cell extraction
cell_extraction:
	python3 -u scripts/extract_cells.py \
		--data_path $(DATASET_PATH) \
		--output_folder_path $(WSI_OUT_PATH) \
		--slide_name $(SLIDE)

# Moco training
moco_training:
	python3 -u moco-v3/main_moco.py \
		-b 1024 \
		--epochs 150 \
		--workers 8 \
		--multiprocessing-distributed \
		--dist-url "tcp://localhost:10001" \
		--world-size 1 \
		--rank 0 \
		--tag $(TAG) \
		--n_cell_max 1000000 \
		--list_slides $(SLIDE) \
		--data_path $(WSI_OUT_PATH)/cell_images

# Slide preprocessing
slide_preprocessing:
	python3 -u scripts/process_visium_slides.py \
		--data_folder $(DATASET_PATH)/Visium/$(SLIDE) \
		--slide_folder $(WSI_PATH) \
		--annotation_folder $(WSI_OUT_PATH)/csv_annotation \
		--model_weights $(MOCO_WEIGHTS)

# sCellST training
sCellST_training:
	python3 -u scripts/train_scellst.py \
		--config_path scripts/parameters/config_reg.yaml \
		--tag $(TAG) \
		--path_data $(DATASET_PATH)/Visium/$(SLIDE)

# sCellST inference
sCellST_inference:
	python3 -u scripts/inference_scellst.py \
		--exp_folder_path lightning_exp \
		--path_data $(DATASET_PATH)/Visium/$(SLIDE) \
		--train_data $(SLIDE) \
		--tag $(TAG) \
		--type both

# Clean up
clean:
	rm -rf dataset
