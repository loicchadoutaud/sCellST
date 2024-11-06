import argparse
import os

from scellst.inference import infer_cell, infer_spot, infer_cell_supervised_exp

if __name__ == "__main__":
    # Read arguments
    parser = argparse.ArgumentParser(description="Mil despot")
    parser.add_argument(
        "-c",
        "--exp_folder_path",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "--train_data",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-p",
        "--path_data",
        default=None,
        type=str,
        help="exp",
    )
    parser.add_argument(
        "--type",
        default=None,
        type=str,
        help="type of inference to make.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        type=str,
        help="tag for output path",
    )
    args = parser.parse_args()

    if args.tag is not None:
        args.exp_folder_path = str(os.path.join(args.exp_folder_path, args.tag))
        output_path = os.path.join("outputs", "adata", args.tag)
    else:
        output_path = os.path.join("outputs", "adata", "other")
    os.makedirs(output_path, exist_ok=True)

    # Choose model
    list_exp = [f for f in os.listdir(args.exp_folder_path)]
    for exp_path in list_exp:
        exp_path = os.path.join(args.exp_folder_path, exp_path)
        # Perform inference
        match args.type:
            case "spot":
                infer_spot(exp_path, args.path_data, output_path)
            case "cell":
                infer_cell(exp_path, args.path_data, output_path)
            case "both":
                infer_spot(exp_path, args.path_data, output_path)
                infer_cell(exp_path, args.path_data, output_path)
            case "supervised":
                infer_cell_supervised_exp(exp_path, args.path_data, output_path)
            case _:
                raise ValueError()

    print("Experiment done.")