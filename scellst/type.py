from enum import Enum


class TaskType(str, Enum):
    regression = "regression"
    nb_total_regression = "nb_total_regression"
    nb_mean_regression = "nb_mean_regression"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ModelType(str, Enum):
    instance_mil = "instance_mil"
    supervised = "supervised"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class PredictorType(str, Enum):
    gene_predictor = "gene_predictor"
    gene_distribution_predictor = "gene_distribution_predictor"
    supervised_predictor = "supervised_predictor"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class MilType(str, Enum):
    instance = "instance"
    embedding = "embedding"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class InputType(str, Enum):
    embedding = "embedding"
    cell_type = "cell_type"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
