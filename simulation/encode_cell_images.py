import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from numpy import ndarray
from torch.utils.data import ConcatDataset
from torchvision.transforms.v2 import Transform, Compose
from tqdm.auto import tqdm

from data_embedder.utils import CustomImageFolderH5


def get_transform(imagesize: int, mean: list, std: list) -> transforms:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(size=imagesize),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_models(
    archi: str, checkpoint: str, mean: list, std: list, image_size: int
) -> tuple[nn.Module, transforms]:
    if archi == "resnet50":
        model = get_moco_resnet(archi, checkpoint)
        transforms = get_transform(image_size, mean, std)
    elif archi == "resnet18":
        model = get_moco_resnet(archi, checkpoint)
        transforms = get_transform(image_size, mean, std)
    elif archi == "dinov2_vitb14":
        model = get_dinov2_vit(archi, checkpoint)
        transforms = get_transform(224, mean, std)
    else:
        raise NotImplementedError
    return model, transforms


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_moco_resnet(modelname: str, checkpoint_path: str) -> nn.Module:
    model = torchvision_models.__dict__[modelname](weights=None)
    if os.path.isfile(checkpoint_path):
        print("Loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not k.startswith(
                "module.base_encoder.%s" % "fc"
            ):
                # remove prefix
                state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % "fc", "%s.bias" % "fc"}

        print("Loaded pre-trained model '{}'".format(checkpoint_path))
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
    model.fc = Identity()
    return model


def get_dinov2_vit(modelname, checkpoint=None):
    model = torch.hub.load("facebookresearch/dinov2", modelname)

    # pos_embed has wrong shape
    if checkpoint is not None:
        pretrained = torch.load(checkpoint, map_location=torch.device("cpu"))["model"]
        # make correct state dict for loading
        new_state_dict = {}
        print(pretrained.keys())
        for key, value in pretrained.items():
            if "dino" in key or "ibot" in key or "student" in key:
                pass
            else:
                print(key)
                new_key = key.replace("teacher.backbone.", "")
                new_state_dict[new_key] = value
        # change shape of pos_embed
        input_dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        pos_embed = nn.Parameter(torch.zeros(1, 257, input_dims[modelname]))
        model.pos_embed = pos_embed
        # load state dict
        model.load_state_dict(new_state_dict, strict=True)
        print("Loaded pre-trained model '{}'".format(checkpoint))
    else:
        print("No checkpoint found at '{}'".format(checkpoint))
    return model


def create_dataset(
    data_path: str, list_slides: list[str], transform: Transform | Compose
) -> ConcatDataset:
    """Create a single dataset with multiple slides."""
    all_datasets = []
    for slide in list_slides:
        all_datasets.append(
            CustomImageFolderH5(os.path.join(data_path, slide), transform)
        )
    return ConcatDataset(all_datasets)


def save_dataset_h5(output_dir: str, embeddings: ndarray) -> None:
    print("Saving embeddings h5...")
    h5file = h5py.File(output_dir, "w")
    h5file.create_dataset(
        "embeddings",
        embeddings.shape,
        dtype=embeddings.dtype,
        data=embeddings,
    )
    print("Done...")


def embed_cells(
    cell_folder_path: str,
    list_tag: list[str],
    model_name: str,
    output_folder_path: str,
    exp_folder_path: str,
    image_size: int = 48,
    batch_size: int = 1024,
    workers: int = 6,
) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Repeat for all tags
    for tag in list_tag:
        # Prepare output dir
        slide_name = os.path.basename(cell_folder_path)

        # Load normalization json file from args.output_dir + args.normalization
        mean_std_path = os.path.join(
            exp_folder_path, f"moco_{tag}_model_best_mean_std.json"
        )
        with open(mean_std_path, "r") as f:
            mean_std = json.load(f)
            mean = mean_std["mean"]
            std = mean_std["std"]

        # Load model
        checkpoint = os.path.join(exp_folder_path, f"moco_{tag}_model_best.pth.tar")
        model, model_transformation = get_models(
            model_name, checkpoint, mean, std, image_size
        )
        model.to(device)
        print("Pushed model to device")

        # Data loading code
        dataset = CustomImageFolderH5(
            cell_folder_path + ".h5",
            model_transformation,
        )
        print("Dataset created")
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # Inference
        print("Starting inference...")
        model.eval()
        idx_start = 0
        embeddings = np.empty((len(dataset), 2048))

        with torch.no_grad():
            for i, batch in tqdm(
                enumerate(loader), total=len(loader), desc="Embedding computations"
            ):
                images = batch.cuda()
                output = model(images)
                embeddings[idx_start : idx_start + len(images)] = output.cpu().numpy()
                idx_start += len(images)

        # h5 dataset
        output_dir = os.path.join(output_folder_path, slide_name)
        os.makedirs(output_dir, exist_ok=True)
        save_dataset_h5(os.path.join(output_dir, tag + ".h5"), embeddings)


if __name__ == "__main__":
    model_names = ["resnet50", "resnet18", "dinov2_vitb14"]

    parser = argparse.ArgumentParser(
        description="Encode cell images using a pretrained model"
    )
    parser.add_argument("--cell_folder_path", type=str)
    parser.add_argument("--list_tag", nargs="+")
    parser.add_argument(
        "--exp_folder_path",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--arch",
        default="resnet50",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=1024,
        type=int,
    )
    args = parser.parse_args()
    embed_cells(
        cell_folder_path=args.cell_folder_path,
        list_tag=args.list_tag,
        model_name=args.arch,
        output_folder_path=args.output_folder_path,
        exp_folder_path=args.exp_folder_path,
    )
