import os

import torch
from loguru import logger
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torchvision.models import resnet18, resnet50


def get_model(model: str) -> nn.Module:
    """Get backbone model from architecture name."""
    if model == "resnet18":
        return resnet18(weights=None)
    elif model == "resnet50":
        return resnet50(weights=None)
    else:
        raise ValueError(f"Unknown architecture: {model}")


def select_only_relevant_weights(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Select only relevant weights for the model. Useful to load weights from a different model trained with moco."""
    for key in list(state_dict.keys()):
        if key.startswith("module.base_encoder") and not key.startswith(
            "module.base_encoder.%s" % "fc"
        ):
            # remove prefix
            state_dict[key[len("module.base_encoder.") :]] = state_dict[key]
        # delete renamed or unused k
        del state_dict[key]
    return state_dict


class InstanceEmbedder(nn.Module):
    """
    Instance embedder for cell images.

    Args:
        archi: Architecture of the model.
        weights: Weights of the model.
    """

    def __init__(
        self,
        archi: str,
        weights: str = "imagenet",
    ):
        super().__init__()
        if "imagenet" not in weights:
            assert os.path.exists(weights), f"{weights} does not exist."
            logger.info(f"Loading pretrained {archi} with {os.path.basename(weights)}")
            pretrained_model = get_model(archi)
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)[
                "state_dict"
            ]
            state_dict = select_only_relevant_weights(state_dict)
            msg = pretrained_model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % "fc", "%s.bias" % "fc"}
            logger.info(f"Loaded pretrained {archi} with {os.path.basename(weights)}")
        else:
            if archi == "resnet18":
                logger.info("Loading pretrained resnet18")
                pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif archi == "resnet50":
                logger.info("Loading pretrained resnet50")
                pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                raise ValueError(f"Unknown architecture: {archi}")
        self.last_dim = pretrained_model.inplanes
        self.model = torch.nn.Sequential(
            *(list(pretrained_model.children())[:-1]) + [torch.nn.Flatten()]
        )

    def get_output_dim(self) -> int:
        return self.last_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
