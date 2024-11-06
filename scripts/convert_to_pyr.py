"""Code imported from https://github.com/mahmoodlab/HEST"""
import argparse

from data_embedder.utils_from_hest import convert_wsi_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a directory of WSIs to pyramidal tiff")
    parser.add_argument("--input_folder_path", type=str, help="Path to the folder containing the WSIs")
    parser.add_argument("--adata_folder_path", type=str, help="Path to the folder containing the visium folders")
    args = parser.parse_args()
    convert_wsi_directory(args.input_folder_path, args.adata_folder_path)

