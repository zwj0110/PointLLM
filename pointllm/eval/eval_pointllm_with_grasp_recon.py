# -*- coding: utf-8 -*-
"""
eval_pointllm_with_grasp_recon.py

Baseline:
  gt points -> PointLLM
  grasp encode+decode -> PointLLM
  zero points -> PointLLM

You only need to:
1) fill in build_grasp_model() with your project's GRASP loader
2) fill in run_pointllm_scoring() with your existing eval logic (already works for "compressed points -> PointLLM")
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# --------------------- project path inject (adjust if needed) ---------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # change if this script is elsewhere
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# --------------------- YOU ALREADY HAVE THESE IN YOUR PROJECT ---------------------
# Example imports (adjust to your actual structure)
# from pointllm.model.pointllm_internal import PointLLMLlamaForCausalLMInternal
# from pointllm.data import make_object_point_data_module
# from transformers import AutoTokenizer

from pointllm.grasp_bridge import GraspEncodeDecodeBridge, GraspBridgeConfig


def build_grasp_model(grasp_config: str, grasp_ckpt: str, device: str = "cuda") -> torch.nn.Module:
    """
    TODO: Replace with your real GRASP builder.

    You previously had something like:
      - load yaml grasp_dus1.yaml
      - build net
      - load r03.pth

    Return a torch.nn.Module that can do encode/decode (or forward) in eval mode.
    """
    raise NotImplementedError(
        "Please implement build_grasp_model() using your existing GRASP loading code."
    )


def run_pointllm_scoring(points_BNx3: torch.Tensor, meta: Dict[str, Any], args) -> Dict[str, Any]:
    """
    TODO: Replace with your working PointLLM eval path.

    Inputs:
      points_BNx3: (B,N,3) float32
      meta: any extra fields from dataset (label, id, text prompt, etc.)
    Output:
      dict with at least:
        - "pred" (int or str)
        - "label" (int or str)
        - optional: logits stats, etc.
    """
    raise NotImplementedError(
        "Please paste your existing PointLLM evaluation forward/scoring here."
    )


def main():
    parser = argparse.ArgumentParser()
    # dataset / eval
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--subset_nums", type=int, default=0)  # 0 means full
    parser.add_argument("--output_json", type=str, default="eval_grasp_pointllm_results.json")

    # modes
    parser.add_argument("--input_mode", type=str, default="grasp",
                        choices=["gt", "grasp", "zero"],
                        help="gt: raw points; grasp: grasp recon points; zero: zero points")

    # grasp
    parser.add_argument("--use_grasp", action="store_true")
    parser.add_argument("--grasp_config", type=str, default="")
    parser.add_argument("--grasp_ckpt", type=str, default="")
    parser.add_argument("--grasp_norm", type=str, default="unit_sphere", choices=["unit_sphere", "unit_cube", "none"])
    parser.add_argument("--grasp_voxel_size", type=float, default=0.0,
                        help="If GRASP RunsenXu_graspnet_enc_dec_r04 is discrete coords, set >0 for dequantization")
    parser.add_argument("--device", type=str, default="cuda")

    # you can add PointLLM args here as needed (base_model, checkpoint, prompt index, etc.)
    args = parser.parse_args()

    # --------------------- build dataset (use your existing datamodule) ---------------------
    # TODO: Replace with your project's dataset builder (ModelNet40 / object point dataset)
    #
    # Example:
    # data_module = make_object_point_data_module(...)
    # test_dataset = data_module["test_dataset"]
    # loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    #
    # For now we raise to force you to paste yours.
    raise NotImplementedError("Please plug in your dataset + DataLoader here.")


if __name__ == "__main__":
    main()
