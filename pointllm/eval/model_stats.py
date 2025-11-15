# pointllm/eval/model_stats.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn

# fvcore is used for FLOPs analysis; install with: pip install fvcore
try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None


# ---------------------------
# Device sync helpers (CUDA/MPS/CPU)
# ---------------------------
def _device_sync():
    """Synchronize device for fair timing when needed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


# ---------------------------
# Parameter counting
# ---------------------------
def params_by_module(model_core: nn.Module) -> Tuple[List[Tuple[str, str, int]], int]:
    """
    Count parameters module-by-module (own parameters only; no recursion) and total.

    Returns:
        rows: List of tuples (module_name, module_class, num_params)
        total_params: int
    """
    rows: List[Tuple[str, str, int]] = []
    total = 0
    for name, mod in model_core.named_modules():
        own_params = sum(p.numel() for p in mod.parameters(recurse=False))
        if own_params > 0:
            rows.append((name, mod.__class__.__name__, own_params))
            total += own_params
    rows.sort(key=lambda x: -x[2])
    return rows, total


# ---------------------------
# FLOPs analysis (single forward)
# ---------------------------
class _ForwardWrapper(nn.Module):
    """
    Wrap the core module to normalize the forward signature so that fvcore can trace it.

    We first try:  forward(input_ids=..., point_clouds=..., use_cache=False)
    If it fails, fall back to: forward(input_ids=..., points=..., use_cache=False)
    (Some repos name the point tensor 'points' instead of 'point_clouds'.)
    """
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, input_ids: torch.Tensor, point_clouds: torch.Tensor):
        kwargs = dict(input_ids=input_ids, use_cache=False)
        # Try common kw name 'point_clouds'
        try:
            out = self.core(**kwargs, point_clouds=point_clouds)
        except TypeError:
            # Fallback to 'points'
            out = self.core(**kwargs, points=point_clouds)
        # fvcore only needs to trace ops; return a tensor-ish output
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        return out


def flops_by_module(
    model_core: nn.Module,
    input_ids: torch.Tensor,
    point_clouds: torch.Tensor
) -> Tuple[Dict[str, int], int]:
    """
    Compute FLOPs for a single forward pass using fvcore.
    The FLOPs reported correspond to ONE forward (not the token-by-token generate loop).

    Returns:
        by_module: dict { "module_name": flops(int) }
        total: int total flops
    """
    if FlopCountAnalysis is None:
        raise RuntimeError("fvcore not available. Please: pip install fvcore")

    wrapper = _ForwardWrapper(model_core).eval()
    with torch.no_grad():
        _device_sync()
        flops = FlopCountAnalysis(wrapper, (input_ids, point_clouds))
        by_module = flops.by_module()
        total = flops.total()
        _device_sync()
    return by_module, total


# ---------------------------
# Grouping / pretty print
# ---------------------------
def group_sum(d: Dict[str, int], prefixes: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Aggregate values by module name prefixes.

    Example:
        prefixes = {
            "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
            "Projector"    : ["point_proj", "mm_projector", "projector"],
            "LLM"          : ["language_model", "model", "transformer", "lm"],
        }
    """
    sums = defaultdict(int)
    for name, val in d.items():
        for group, pres in prefixes.items():
            if any(name.startswith(p) for p in pres):
                sums[group] += val
                break
    return dict(sums)


def pretty_print_params(rows: List[Tuple[str, str, int]], total: int, topk: int = 30) -> None:
    print("\n===== #Params by Module =====")
    shown = rows[:topk]
    wname = max((len(n) for n, _, _ in shown), default=10)
    for name, cls, nparam in shown:
        print(f"{name:<{wname}}  {cls:>24}  {nparam/1e6:8.3f} M")
    print(f"TOTAL: {total/1e6:.3f} M")
    print("=============================\n")
def _human_flops(x: float):
    """Return value and unit with adaptive scaling."""
    ax = abs(x)
    if ax >= 1e12:  return x / 1e12, "TFLOPs"
    if ax >= 1e9:   return x / 1e9, "GFLOPs"
    if ax >= 1e6:   return x / 1e6, "MFLOPs"
    if ax >= 1e3:   return x / 1e3, "KFLOPs"
    return x, "FLOPs"

def pretty_print_flops(by_module: Dict[str, float], total: float, topk: int = 30) -> None:
    """
    by_module/total 的单位应是“原始 FLOPs”（未除以 1e9）。
    依赖外部的 `_human_flops(x) -> (val, unit)`。
    """
    # 防御：如果 total 传 0 或 None，就用 by_module 求和
    total_flops = float(total) if total else float(sum(by_module.values()))

    # 排序并截取 topk
    items = sorted(by_module.items(), key=lambda x: x[1], reverse=True)[:topk]
    wname = max((len(k) for k, _ in items), default=10)

    # 打印头
    tot_val, tot_unit = _human_flops(total_flops)
    print("\n===== FLOPs by Module (single forward) =====")
    print(f"{'Total:':<{wname}}  {tot_val:>12.3f} {tot_unit:7s}  (100.000%)")

    # 每项：自适应单位 + 占比；小项用科学计数法避免 0.000
    for k, v in items:
        val, unit = _human_flops(v)
        share = (v / total_flops * 100.0) if total_flops > 0 else 0.0
        sval = f"{val:.3f}" if abs(val) >= 1e-3 else f"{val:.3e}"
        print(f"{k:<{wname}}  {sval:>12s} {unit:7s}  ({share:6.3f}%)")

    print("===========================================\n")



# ---------------------------
# Optional: CSV dump helpers
# ---------------------------
def dump_params_csv(rows: List[Tuple[str, str, int]], total: int, path: str) -> None:
    """
    Dump per-module params and total to CSV (module, class, params).
    """
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "class", "params"])
        for name, cls, nparam in rows:
            w.writerow([name, cls, nparam])
        w.writerow(["TOTAL", "", total])


def dump_flops_csv(by_module: Dict[str, int], total: int, path: str) -> None:
    """
    Dump per-module FLOPs and total to CSV (module, flops).
    """
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "flops"])
        for k, v in sorted(by_module.items(), key=lambda x: -x[1]):
            w.writerow([k, v])
        w.writerow(["TOTAL", total])


# ---------------------------
# Quick demo helpers (optional)
# ---------------------------
def quick_show_params(core: nn.Module, topk: int = 40) -> Dict[str, float]:
    """
    Quickly print and return grouped params (Point Encoder / Projector / LLM).
    You may adjust prefixes to match your repo's module names.
    """
    rows, total = params_by_module(core)
    pretty_print_params(rows, total, topk=topk)

    # adjust prefixes according to your model tree
    prefixes = {
        "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
        "Projector"    : ["point_proj", "mm_projector", "projector"],
        "LLM"          : ["language_model", "model", "transformer", "lm"],
    }
    grouped = group_sum({name: n for name, _, n in rows}, prefixes)
    print("Grouped Params (M):", {k: v/1e6 for k, v in grouped.items()})
    return {k: v/1e6 for k, v in grouped.items()}


def quick_show_flops(core: nn.Module, input_ids: torch.Tensor, point_clouds: torch.Tensor,
                     topk: int = 40) -> Dict[str, float]:
    """
    Quickly print and return grouped FLOPs for a single forward.
    """
    if FlopCountAnalysis is None:
        raise RuntimeError("fvcore not available. Please: pip install fvcore")
    by_mod, total = flops_by_module(core.eval(), input_ids, point_clouds)
    pretty_print_flops(by_mod, total, topk=topk)

    prefixes = {
        "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
        "Projector"    : ["point_proj", "mm_projector", "projector"],
        "LLM"          : ["language_model", "model", "transformer", "lm"],
    }
    grouped = group_sum(by_mod, prefixes)
    print("Grouped FLOPs (GFLOPs, single forward):", {k: v/1e9 for k, v in grouped.items()})
    return {k: v/1e9 for k, v in grouped.items()}
