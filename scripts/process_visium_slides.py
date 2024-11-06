import argparse

from data_embedder.visium_processor import VisiumProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visium preprocessing")
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        help="data folder path (default: None)",
    )
    parser.add_argument(
        "-s",
        "--slide_folder",
        type=str,
        help="slide folder path (default: None)",
    )
    parser.add_argument(
        "-f",
        "--annotation_folder",
        type=str,
        help="slide folder path (default: None)",
    )
    parser.add_argument(
        "-w",
        "--model_weights",
        type=str,
        help="name of file weights to use",
    )
    parser.add_argument(
        "-i",
        "--img_size",
        default=12.,
        type=float,
        help="Image size in micrometers",
    )
    args = parser.parse_args()

    print(f"Processing {args.data_folder}...")

    # Perform preprocessing
    visium_slide = VisiumProcessor(
        input_folder_path=args.data_folder,
        slide_folder_path=args.slide_folder,
        annotation_folder_path=args.annotation_folder,
        img_size=args.img_size,
        radius_ratio=1.0,
        model_name="resnet50",
        model_weights=args.model_weights,
    )
    visium_slide.process_image()
    print(f"File {args.data_folder} processed !")

