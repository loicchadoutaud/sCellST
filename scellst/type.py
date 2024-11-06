from enum import Enum


class TaskType(str, Enum):
    regression = "regression"
    nb_total_regression = "nb_total_regression"
    nb_mean_regression = "nb_mean_regression"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class AggType(str, Enum):
    mean = "mean"
    sum = "sum"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class ModelType(str, Enum):
    instance_mil = "instance_mil"
    supervised = "supervised"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class InputType(str, Enum):
    embedding = "embedding"
    cell_type = "cell_type"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
