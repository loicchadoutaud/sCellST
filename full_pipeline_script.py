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
