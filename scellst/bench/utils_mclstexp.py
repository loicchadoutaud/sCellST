import argparse
import os
import shutil
import sys
from pathlib import Path

from lightning import seed_everything

sys.path.insert(0, os.path.abspath("external/mclSTExp"))

import scanpy as sc
import torch
import numpy as np
import torch.nn.functional as F
from anndata import AnnData
from omegaconf import OmegaConf
from loguru import logger
from tqdm.auto import tqdm
from scellst.constant import MODELS_DIR, METRICS_DIR
from scellst.dataset.data_module import prepare_data_module
from scellst.io_utils import load_config, load_yaml
from scellst.utils import update_config, create_tag
from scellst.metrics.gene import compute_gene_metrics

from external.mclSTExp.model import mclSTExp_Attention
from external.mclSTExp.train import train

MODEL_NAME = "MclSTExp"


def get_default_args():
    return argparse.Namespace(
        batch_size=128,
        max_epochs=90,
        temperature=1.0,
        fold=0,
        dim=None,
        image_embedding_dim=1024,
        projection_dim=256,
        heads_num=8,
        heads_dim=64,
        heads_layers=2,
        dropout=0.0,
        encoder_name="densenet121",
    )


def find_matches(spot_embeddings, query_embeddings, top_k=1):
    # find the closest matches
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T
    logger.info(f"Similarity shape: {dot_similarity.shape}")
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)

    return indices.cpu().numpy()


def get_embeddings_adapted(model_path, model, test_loader) -> AnnData:
    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace("module.", "")  # remove the prefix 'module.'
        new_key = new_key.replace("well", "spot")  # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to("cuda")
    logger.info("Finished loading model")

    all_img_embeddings = []
    all_spot_embeddings = []
    all_spot_expression = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            all_img_embeddings.append(image_embeddings)

            all_spot_expression.append(batch["expression"])
            spot_feature = batch["expression"].cuda()
            x = batch["position"][:, 0].long().cuda()
            y = batch["position"][:, 1].long().cuda()
            centers_x = model.x_embed(x)
            centers_y = model.y_embed(y)
            spot_feature = spot_feature + centers_x + centers_y
            spot_features = spot_feature.unsqueeze(dim=0)
            spot_embedding = model.spot_encoder(spot_features)
            spot_embedding = model.spot_projection(spot_embedding).squeeze(dim=0)
            all_spot_embeddings.append(spot_embedding)

    all_img_embeddings, all_spot_embeddings = torch.cat(all_img_embeddings), torch.cat(
        all_spot_embeddings
    )
    X = torch.cat(all_spot_expression)
    return AnnData(
        X=X.numpy(),
        obsm={
            "img_embeddings": all_img_embeddings.cpu().numpy(),
            "spot_embeddings": all_spot_embeddings.cpu().numpy(),
        },
    )


def compute_average_expression(
    adata_key: AnnData, indices: np.ndarray, adata_query: AnnData
) -> AnnData:
    matched_spot_embeddings_pred = np.zeros(
        (indices.shape[0], adata_key.obsm["spot_embeddings"].shape[1])
    )
    matched_spot_expression_pred = np.zeros((indices.shape[0], adata_key.X.shape[1]))
    for i in range(indices.shape[0]):
        a = np.linalg.norm(
            adata_key.obsm["spot_embeddings"][indices[i, :], :]
            - adata_query.obsm["img_embeddings"][i, :],
            axis=1,
        )
        reciprocal_of_square_a = np.reciprocal(a**2)
        weights = reciprocal_of_square_a / np.sum(reciprocal_of_square_a)
        weights = weights.flatten()
        matched_spot_embeddings_pred[i, :] = np.average(
            adata_key.obsm["spot_embeddings"][indices[i, :], :], axis=0, weights=weights
        )
        matched_spot_expression_pred[i, :] = np.average(
            adata_key.X[indices[i, :], :], axis=0, weights=weights
        )
    logger.info(
        f"matched spot embeddings pred shape: {matched_spot_embeddings_pred.shape}",
    )
    logger.info(
        f"matched spot expression pred shape: {matched_spot_expression_pred.shape}"
    )
    return AnnData(
        X=matched_spot_expression_pred, obs=adata_query.obs, var=adata_key.var
    )


def train_mclstexp(config_path: Path, config_kwargs: dict) -> None:
    logger.info(f"Training: {MODEL_NAME}")

    # Load config
    config = load_config(config_path)
    config = update_config(config, config_kwargs)
    config.exp_tag = create_tag(config_kwargs)

    # Set seed
    seed_everything(config.data.seed)

    # Load method args
    args = get_default_args()
    config.data.batch_size = args.batch_size

    # Load data
    data_module = prepare_data_module(
        config.data, stage="fit", task_type=config.model.task_type
    )

    # Prepare model
    config.model.gene_names = data_module.get_gene_names()
    args.dim = len(data_module.get_gene_names())
    model = mclSTExp_Attention(
        encoder_name=args.encoder_name,
        spot_dim=args.dim,
        temperature=args.temperature,
        image_dim=args.image_embedding_dim,
        projection_dim=args.projection_dim,
        heads_num=args.heads_num,
        heads_dim=args.heads_dim,
        head_layers=args.heads_layers,
        dropout=args.dropout,
    )
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    output_dir = MODELS_DIR / MODEL_NAME / config.exp_tag

    # Train model
    for epoch in range(args.max_epochs):
        model.train()
        train(model, data_module.train_dataloader(), optimizer, epoch)

    # Save model
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best.pt")
    OmegaConf.save(config=config, f=output_dir / "config.yaml")

    # Save spot embeddings
    adata = get_embeddings_adapted(
        model_path=output_dir / "best.pt",
        model=model,
        test_loader=data_module.train_dataloader(),
    )
    adata.var_names = config.model.gene_names
    adata.write(output_dir / "adata_key.h5ad")
    logger.info("Saved Model")


def eval_mclstexp(config_dir: Path, config_kwargs: dict) -> None:
    logger.info(f"Evaluating: {MODEL_NAME}")

    # Setup config
    config = load_yaml(config_dir / "config.yaml")
    config = update_config(config, config_kwargs)
    config.data.genes = config.model.gene_names

    # Set seed
    seed_everything(config.data.seed)

    # Load method args
    args = get_default_args()

    # Load data
    data_module = prepare_data_module(
        config.data, stage="predict", task_type=config.model.task_type
    )

    # Prepare model
    output_dir = MODELS_DIR / MODEL_NAME / config.exp_tag
    logger.info(f"Loading trained model from {output_dir}")
    args.dim = len(data_module.get_gene_names())
    model = mclSTExp_Attention(
        encoder_name=args.encoder_name,
        spot_dim=args.dim,
        temperature=args.temperature,
        image_dim=args.image_embedding_dim,
        projection_dim=args.projection_dim,
        heads_num=args.heads_num,
        heads_dim=args.heads_dim,
        head_layers=args.heads_layers,
        dropout=args.dropout,
    )

    # Evaluate model
    adata_truth = get_embeddings_adapted(
        model_path=output_dir / "best.pt",
        model=model,
        test_loader=data_module.predict_dataloader(),
    )
    adata_truth.var_names = config.model.gene_names

    # Load trained key
    logger.info(f"Loading key anndata from {output_dir / 'adata_key.h5ad'}")
    adata_key = sc.read_h5ad(output_dir / f"adata_key.h5ad")

    indices = find_matches(
        adata_key.obsm["spot_embeddings"], adata_truth.obsm["img_embeddings"], top_k=200
    )
    adata_pred = compute_average_expression(adata_key, indices, adata_truth)
    adata_pred.obsm["spatial"] = data_module.adata.obsm["spatial"]
    adata_pred.uns = data_module.adata.uns
    logger.info(f"Predicted {adata_pred.shape} / {adata_truth.shape} spots.")

    # Compute metrics
    metrics = compute_gene_metrics(adata_truth, adata_pred)
    output_dir = METRICS_DIR / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{config.exp_tag};test_slide={config.data.predict_id};model={MODEL_NAME}.csv"
    )
    metrics.to_csv(output_path)

    # # Plotting
    # config.save_dir_tag = "benchmark"
    # config.infer_mode = "spot"
    # save_plots(metrics, data_module.adata, adata_pred, config)
