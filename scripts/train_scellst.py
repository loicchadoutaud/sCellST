import os
import argparse
from functools import partial

from scellst.config import Config
from scellst.train import train_supervised
from scellst.type import ModelType
from scellst.utils import read_yaml, save_yaml, prepare_configs, prepare_configs_from_list
from scellst import train


if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser(description="Mil despot")
    parser.add_argument(
        "-c",
        "--config_path",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-p",
        "--path_data",
        default=None,
        help="data folder path (default: None)",
    )
    parser.add_argument(
        "-t",
        "--tag",
        default=None,
        help="tag experiments",
    )
    args = parser.parse_args()

    # Run main script
    template_conf = read_yaml(args.config_path)
    template_conf["train_data_folder"] = os.path.basename(args.path_data)
    template_conf["trainer"]["exp_id"] = "demo"
    template_conf["data"]["tag_ssl"] = "demo"

    # Option for this experiments
    os.makedirs("lightning_exp", exist_ok=True)
    exp_folder_path = os.path.join("lightning_exp", args.tag)

    os.makedirs(os.path.join(exp_folder_path, template_conf["trainer"]["exp_id"]), exist_ok=True)
    save_yaml(os.path.join(exp_folder_path, template_conf["trainer"]["exp_id"], "parameters.yaml"), template_conf)

    # Run experiment
    train(Config(**template_conf), args.path_data, exp_folder_path)

    print("Experiment done.")
