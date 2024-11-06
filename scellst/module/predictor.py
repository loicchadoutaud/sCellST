from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import nn, Tensor
from torch.nn.functional import linear, one_hot

from scellst import REGISTRY_KEYS
from scellst.module.distributions import NegativeBinomial


class BasePredictor(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        final_activation: str = "softplus",
        hidden_dim: Optional[List[int]] = None,
        dropout_rate: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else []
        self.dropout_rate = dropout_rate
        if final_activation == "relu":
            self.final_activation_layer = nn.ReLU()
        elif final_activation == "softplus":
            self.final_activation_layer = nn.Softplus(beta=20)
        elif final_activation == "sigmoid":
            self.final_activation_layer = nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation_layer = nn.Softmax(dim=-1)
        elif final_activation == "identity":
            self.final_activation_layer = nn.Identity()
        else:
            raise ValueError(
                f"final activation layer must be one of [relu, softplus, sigmoid, softmax, identity], got {final_activation}"
            )

        self.input_dims = [input_dim] + self.hidden_dim
        self.output_dims = self.hidden_dim + [output_dim]

        self.model_latent = self._create_latent_model()
        self.model_out = self._create_out_model()

    def _create_latent_model(self) -> nn.Module:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_channel, out_channel),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                )
                for in_channel, out_channel in zip(
                    self.input_dims[:-1], self.output_dims[:-1]
                )
            ],
        )

    def _create_out_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_dims[-1], self.output_dims[-1]),
            self.final_activation_layer,
        )

    def get_latent_dim(self) -> int:
        return self.input_dims[-1]

    @abstractmethod
    def forward(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor | NegativeBinomial]:
        pass


class Predictor(BasePredictor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        final_activation: str = "softplus",
        hidden_dim: Optional[List[int]] = None,
        dropout_rate: Optional[float] = 0.0,
    ):
        super().__init__(
            input_dim,
            output_dim,
            final_activation,
            hidden_dim,
            dropout_rate,
        )

    def forward(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor | NegativeBinomial]:
        h = self.model_latent(batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING])
        out = self.model_out(h)
        return {
            REGISTRY_KEYS.OUTPUT_PREDICTION: out,
            REGISTRY_KEYS.OUTPUT_LATENT: h,
        }


class DistributionPredictor(BasePredictor):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dispersion: str = "gene",
        final_activation: str = "softplus",
        hidden_dim: Optional[List[int]] = None,
        dropout_rate: Optional[float] = 0.0,
    ):
        super().__init__(
            input_dim,
            output_dim,
            final_activation,
            hidden_dim,
            dropout_rate,
        )
        # Parameters in case of nb models
        self.dispersion = dispersion
        if dispersion == "gene":
            self.gene_param = torch.nn.Parameter(torch.randn(self.output_dim))
        else:
            raise ValueError(
                f"dispersion must be either: gene or gene-batch, got {dispersion}"
            )

    def forward(
        self, batch_data: dict[str, Tensor]
    ) -> dict[str, Tensor | NegativeBinomial]:
        # Stack inputs
        stacked_instance_embeddings = batch_data[REGISTRY_KEYS.OUTPUT_EMBEDDING]

        # Compute negative binomial parameters
        h = self.model_latent(stacked_instance_embeddings)

        # Get gene params
        if self.dispersion == "gene":
            gene_param = self.gene_param.repeat((len(stacked_instance_embeddings), 1))
        else:
            raise ValueError("This should not happen.")
        return {
            REGISTRY_KEYS.OUTPUT_PREDICTION: self.model_out(h),
            REGISTRY_KEYS.OUTPUT_GENE_PARAM: gene_param,
            REGISTRY_KEYS.OUTPUT_LATENT: h,
        }
