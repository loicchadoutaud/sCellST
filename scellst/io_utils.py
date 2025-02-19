import os
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf, DictConfig
from pandas import DataFrame


def load_config(config_path: str | Path) -> DictConfig:
    # Load the main configuration file
    config = OmegaConf.load(config_path)

    # Resolve the defaults list
    defaults = config.get("defaults", [])
    base_dir = os.path.dirname(config_path)

    for default in defaults:
        for key, value in default.items():
            default_path = os.path.join(base_dir, key, f"{value}.yaml")
            default_config = OmegaConf.load(default_path)
            config = OmegaConf.merge(config, default_config)

    # Remove the defaults key from the final configuration
    if "defaults" in config:
        del config["defaults"]

    return config


def load_yaml(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def load_multirun(path_dir: str) -> tuple[DataFrame, list[str], list[str]]:
    all_metrics = []
    exp_dirs = [
        dir
        for dir in os.listdir(path_dir)
        if os.path.isdir(os.path.join(path_dir, dir))
    ]
    for dir in exp_dirs:
        # Prepare parameters
        parameter_path = os.path.join(path_dir, dir, ".hydra", "overrides.yaml")
        list_params = OmegaConf.load(parameter_path)
        list_params = [param.split("=") for param in list_params]
        for param in list_params:
            if "." in param[0]:
                param[0] = param[0].split(".")[-1]
        param_names = [param[0] for param in list_params]
        param_values = [param[1] for param in list_params]

        # Load metrics
        path_metrics = os.path.join(path_dir, dir, "metrics.json")
        metrics = pd.read_json(path_metrics, orient="records")
        metric_names = metrics.columns
        metrics[param_names] = param_values

        all_metrics.append(metrics)
    all_metrics = pd.concat(all_metrics)

    for param in param_names:
        try:
            all_metrics[param] = pd.to_numeric(all_metrics[param])
        except ValueError:
            pass
    return all_metrics, metric_names, param_names
