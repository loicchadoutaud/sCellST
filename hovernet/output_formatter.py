import glob
import json
import os

import numpy as np
import pandas as pd
from pandas import DataFrame

from constants import TYPE_INFO
from misc.utils import log_info


class OutputFormatter:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.save_dir = os.path.join(self.output_dir, "csv_annotation")
        os.makedirs(self.save_dir, exist_ok=True)

    def _load_results(
        self, wsi_basename: str,
    ) -> DataFrame:
        log_info(f"Loading results...")
        json_path_wsi = os.path.join(self.output_dir, "json",  wsi_basename + ".json")

        min_bbox_list, max_bbox_list, centroid_list = [], [], []
        label, class_label, score = [], [], []

        # add results to individual lists
        with open(json_path_wsi) as json_file:
            data = json.load(json_file)
            nuc_info = data["nuc"]
            for inst in nuc_info:
                inst_info = nuc_info[inst]
                min_bbox_list.append(inst_info["bbox"][0])
                max_bbox_list.append(inst_info["bbox"][1])
                centroid_list.append(inst_info["centroid"])
                label.append(inst_info["type"])
                class_label.append(TYPE_INFO[str(inst_info["type"])][0])
                score.append(inst_info["type_prob"])

        # Format everything into dataframe
        df_annotations = pd.DataFrame(index=np.arange(len(min_bbox_list)))
        df_annotations[["x_min", "y_min"]] = min_bbox_list
        df_annotations[["x_max", "y_max"]] = max_bbox_list
        df_annotations[["x_center", "y_center"]] = centroid_list
        df_annotations["label"] = label
        df_annotations["class"] = df_annotations["label"].map(lambda x:TYPE_INFO[str(x)][0])
        df_annotations["score"] = score
        return df_annotations

    def _format_wsi_output(self, wsi_path: str) -> None:
        wsi_basename = os.path.basename(wsi_path)
        wsi_basename, wsi_ext = os.path.splitext(wsi_basename)
        if os.path.exists(os.path.join(self.output_dir, "json",  wsi_basename + ".json")):
            log_info("Output files found, formatting...")
            # Load results
            df_annotations = self._load_results(wsi_basename)

            # Save outputs
            df_annotations.to_csv(os.path.join(self.save_dir, wsi_basename + '_hovernet_annotations.csv'))
        else:
            log_info("Output files not found, skipping to next slide...")

    def format_wsi_list(self) -> None:
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py

        """
        wsi_path_list = glob.glob(self.input_dir + "/*")
        wsi_path_list.sort()

        for wsi_path in wsi_path_list:
            log_info(f"Formatting outputs for {wsi_path}...")
            self._format_wsi_output(wsi_path)
        log_info("End of formatting.")