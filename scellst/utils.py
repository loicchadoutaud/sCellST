import copy
import logging

import anndata as ad
import numpy as np
import torch
import yaml
from anndata import AnnData
from numpy import ndarray
from sklearn.model_selection import ParameterGrid

from scellst.module.distributions import NegativeBinomial

logger = logging.getLogger(__name__)


def read_yaml(filename: str) -> dict:
    with open(filename) as yaml_file:
        return yaml.safe_load(yaml_file)


def save_yaml(filename: str, dict_to_save: dict) -> None:
    with open(filename, "w") as outfile:
        yaml.dump(dict_to_save, outfile, default_flow_style=False)


def update_conf(conf: dict, list_param: dict) -> dict:
    if "data" in conf.keys():
        # evaluate first condition if it is false don't try to evaluate the other
        if ("gene_to_pred" in list_param.keys()) and len(
            list_param["gene_to_pred"]
        ) > 20:
            list_param.pop("gene_to_pred", None)
    conf["trainer"]["exp_id"] = ";".join(
        [f"train_data-{conf['train_data_folder']}"]
        + [f"{key}-{value}" for key, value in list_param.items()]
    )
    return conf


def find_subdict_keys(searched_key: str, dict_object: dict) -> str:
    for key, value in dict_object.items():
        if isinstance(value, dict):
            if searched_key in value.keys():
                return key
    raise ValueError(f"{searched_key} not found in subdict")


def prepare_config(template: dict, parameter_dict: dict) -> dict:
    conf = copy.deepcopy(template)
    for key, value in parameter_dict.items():
        if not key in conf.keys():
            sub_key = find_subdict_keys(key, template)
            conf[sub_key][key] = value
        else:
            conf[key] = value
    return conf


def prepare_configs(template: dict, hyperparameters: dict) -> list[dict]:
    if "gene_to_pred" in hyperparameters.keys():
        hyperparameters["gene_to_pred"] = [
            sorted(gl) for gl in hyperparameters["gene_to_pred"]
        ][:50]
    list_parameter_dict = list(ParameterGrid(hyperparameters))
    configs = [
        prepare_config(template, parameter_dict)
        for parameter_dict in list_parameter_dict
    ]
    return [
        update_conf(conf, param) for conf, param in zip(configs, list_parameter_dict)
    ]


def prepare_configs_from_list(template: dict, list_parameter_dict: list[dict]) -> list[dict]:
    configs = [
        prepare_config(template, parameter_dict)
        for parameter_dict in list_parameter_dict
    ]
    return [
        update_conf(conf, param) for conf, param in zip(configs, list_parameter_dict)
    ]


def convert_params_to_nb(params: ndarray) -> NegativeBinomial:
    return NegativeBinomial(
        mu=torch.from_numpy(params[:, 0]),
        theta=torch.from_numpy(params[:, 1]),
    )

