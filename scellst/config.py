import logging
from dataclasses import InitVar
from typing import cast

from pydantic import Field
from dataclasses import dataclass

from scellst.type import TaskType, ModelType, InputType

SENTINEL = cast(None, object())


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    normalize: bool
    log1p: bool
    filtering: bool
    radius_ratio: float
    gene_to_pred: list[str]
    gene_type: str
    n_genes: int

    task_type: InitVar[str]

    def __post_init__(self, task_type):
        if task_type in [
            TaskType.nb_total_regression,
            TaskType.nb_mean_regression,
        ]:
            self.log1p = False
            self.normalize = False



@dataclass
class DataConfig:
    task_type: str
    n_folds: int
    batch_size: int
    scale_labels: bool
    dataset_type: str
    model_type: str

    extraction_img_size: InitVar[float]
    archi: InitVar[str]
    weights: InitVar[str]
    tag_ssl: InitVar[str]

    folder_info: str = Field(default="")

    def __post_init__(self, extraction_img_size, archi, weights, tag_ssl):
        match self.dataset_type:
            case InputType.cell_type:
                self.folder_info = ""
            case InputType.embedding:
                if weights == "moco":
                    self.folder_info = f"embedding_{weights}_{tag_ssl}_{archi}_{extraction_img_size}"
                else:
                    self.folder_info = f"embedding_{weights}_{archi}_{extraction_img_size}"
            case _:
                raise ValueError(f"Wrong input type, got: {self.dataset_type}")
        if self.task_type in [
            TaskType.nb_total_regression,
            TaskType.nb_mean_regression,
        ]:
            self.scale_labels = False


@dataclass
class TrainerConfig:
    lr: float
    max_epoch: int
    patience: int
    exp_id: str

@dataclass
class PredictorConfig:
    input_dim: int
    hidden_dim: list[int]
    output_dim: int
    final_activation: str
    dropout_rate: float
    dispersion: str | None = SENTINEL

    def __post_init__(self):
        self.__dataclass_fields__ = {
            k: v for k, v in self.__dataclass_fields__.items()
            if getattr(self, k) is not SENTINEL
        }


@dataclass
class ModelConfig:
    aggregation_type: str | None = SENTINEL
    parametrisation: str | None = SENTINEL

    def __post_init__(self):
        self.__dataclass_fields__ = {
            k: v for k, v in self.__dataclass_fields__.items()
            if getattr(self, k) is not SENTINEL
        }


@dataclass
class Config:
    train_data_folder: str
    task_type: str
    device: str
    model_type: str
    seed: int
    loss: str

    preprocessing: InitVar[dict]
    data: InitVar[dict]
    trainer: InitVar[dict]
    embedder: InitVar[dict]
    predictor: InitVar[dict]
    model: InitVar[dict]

    preprocessing_config: PreprocessingConfig = None
    data_config: DataConfig = None
    trainer_config: TrainerConfig = None
    predictor_config: PredictorConfig = None
    model_config: ModelConfig = None

    def __post_init__(
        self,
        preprocessing,
        data,
        trainer,
        embedder,
        predictor,
        model,
    ):
        # Preprocessing config
        preprocessing.update({"task_type": self.task_type})
        self.preprocessing_config = PreprocessingConfig(**preprocessing)

        # Data config
        data.update(
            {
                "weights": embedder["weights"],
                "archi": embedder["archi"],
                "task_type": self.task_type,
                "model_type": self.model_type,
            }
        )
        if (data["tag_ssl"] is None) & (data["weights"] == "moco"):
            if "Breast" in self.train_data_folder:
                data["tag_ssl"] = "breast"
            elif "DonorA" in self.train_data_folder:
                data["tag_ssl"] = "pdac_donor_A"
            elif "DonorB" in self.train_data_folder:
                data["tag_ssl"] = "pdac_donor_B"
            elif "DonorC" in self.train_data_folder:
                data["tag_ssl"] = "pdac_donor_C"
            elif "Ovarian" in self.train_data_folder:
                data["tag_ssl"] = "ovarian"
            logger.info(f"Setting tag_ssl to {data['tag_ssl']}")
        self.data_config = DataConfig(**data)

        # Predictor and aggregation config
        if self.data_config.dataset_type == "cell_type":
            input_dim = 6
        else:
            input_dim = (
                512 if embedder["archi"] == "resnet18" else 2048
            )
        predictor["input_dim"] = input_dim

        if self.task_type in ([TaskType.nb_total_regression, TaskType.nb_mean_regression]):
            model["parametrisation"] = self.task_type
            self.loss = "nll"
        else:
            del predictor["dispersion"]
        if self.model_type == ModelType.supervised:
            del model["aggregation_type"]
        self.predictor_config = PredictorConfig(**predictor)
        self.model_config = ModelConfig(**model)

        # Trainer config
        self.trainer_config = TrainerConfig(**trainer)