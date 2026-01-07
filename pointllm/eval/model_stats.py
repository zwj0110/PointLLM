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
        # Model should already be in float32 if needed (handled in measure_model_complexity)
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


# ---------------------------
# Comprehensive Model Complexity Analysis
# ---------------------------
def measure_model_complexity(
    model: nn.Module,
    tokenizer,
    point_cloud_shape: Tuple[int, int] = (1, 8192, 3),
    num_warmup: int = 3,
    num_trials: int = 10,
    device: str = None,
    use_adapter: bool = True,
    logger=None
) -> Dict:
    """
    Comprehensive model complexity measurement including:
    - Model size (MB/GB)
    - Trainable parameters (millions/billions)
    - Total parameters (millions/billions)
    - kMACs (kilo Multiply-Accumulate operations)
    - Inference time (ms per point cloud)
    
    Args:
        model: The PointLLM model
        tokenizer: Tokenizer for creating input_ids
        point_cloud_shape: Shape of point cloud (batch, num_points, dims)
        num_warmup: Number of warmup inference runs
        num_trials: Number of trials for timing
        device: Device to run on (auto-detect if None)
        use_adapter: Whether adapter is enabled (for logging purposes)
        logger: Logger instance (uses print if None)
    
    Returns:
        Dictionary with all metrics
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    model = model.to(device)
    model.eval()
    
    # Detect model dtype from first parameter
    model_dtype = next(model.parameters()).dtype
    # Store original dtype to restore later
    original_dtype = model_dtype
    
    # For FLOPs calculation and inference timing, use float32 to avoid dtype mismatch issues
    # This is safe because we're only measuring, not training
    use_float32_for_measurement = model_dtype in (torch.float16, torch.bfloat16)
    
    if use_float32_for_measurement:
        # Temporarily convert model to float32 for measurements
        model = model.float()
        point_cloud_dtype = torch.float32
    else:
        point_cloud_dtype = torch.float32
    
    # Create dummy inputs
    batch_size, num_points, point_dims = point_cloud_shape
    point_clouds = torch.randn(batch_size, num_points, point_dims, device=device, dtype=point_cloud_dtype)
    
    # Create dummy input_ids (simple prompt)
    dummy_text = "What is this point cloud?"
    try:
        input_ids = tokenizer(dummy_text, return_tensors="pt")["input_ids"].to(device)
    except:
        # Fallback: create a simple input_ids tensor
        input_ids = torch.randint(0, 1000, (batch_size, 10), device=device, dtype=torch.long)
    
    results = {}
    
    # ========== 1. Parameter Counting ==========
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results["total_parameters"] = total_params
    results["trainable_parameters"] = trainable_params
    results["total_parameters_M"] = total_params / 1e6
    results["trainable_parameters_M"] = trainable_params / 1e6
    results["total_parameters_B"] = total_params / 1e9
    results["trainable_parameters_B"] = trainable_params / 1e9
    
    # ========== 2. Model Size ==========
    # Calculate model size in bytes (assuming float32/float16)
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024 ** 2)
    total_size_gb = total_size_bytes / (1024 ** 3)
    
    results["model_size_bytes"] = total_size_bytes
    results["model_size_MB"] = total_size_mb
    results["model_size_GB"] = total_size_gb
    
    # ========== 3. FLOPs / MACs ==========
    total_flops = None
    total_macs = None
    flops_by_mod = None
    
    if FlopCountAnalysis is not None:
        try:
            wrapper = _ForwardWrapper(model).eval()
            with torch.no_grad():
                _device_sync()
                flops_analysis = FlopCountAnalysis(wrapper, (input_ids, point_clouds))
                flops_by_mod = flops_analysis.by_module()
                total_flops = flops_analysis.total()
                _device_sync()
            
            # MACs = FLOPs / 2 (one multiply-accumulate = 2 operations)
            total_macs = total_flops / 2
            results["total_FLOPs"] = total_flops
            results["total_MACs"] = total_macs
            results["total_kMACs"] = total_macs / 1e3
            results["total_GMACs"] = total_macs / 1e9
            results["flops_by_module"] = flops_by_mod
        except Exception as e:
            logger.warning(f"Failed to compute FLOPs: {e}")
            results["flops_error"] = str(e)
    else:
        logger.warning("fvcore not available. FLOPs/MACs not computed.")
        results["flops_error"] = "fvcore not installed"
    
    # ========== 4. Inference Time ==========
    # Model is already in float32 if it was converted above
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                _ = model(input_ids=input_ids, point_clouds=point_clouds, use_cache=False)
            except Exception as e:
                logger.debug(f"Warmup inference warning: {e}")
                pass
        _device_sync()
    
    # Timing trials
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            _device_sync()
            start_time = time.perf_counter()
            try:
                _ = model(input_ids=input_ids, point_clouds=point_clouds, use_cache=False)
            except Exception as e:
                logger.warning(f"Inference failed during timing: {e}")
                break
            _device_sync()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Restore original dtype if we converted it
    if use_float32_for_measurement and original_dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype=original_dtype)
    
    if times:
        avg_time_ms = sum(times) / len(times)
        min_time_ms = min(times)
        max_time_ms = max(times)
        results["inference_time_ms"] = avg_time_ms
        results["inference_time_min_ms"] = min_time_ms
        results["inference_time_max_ms"] = max_time_ms
        results["inference_time_std_ms"] = (sum((t - avg_time_ms) ** 2 for t in times) / len(times)) ** 0.5
    else:
        results["inference_time_ms"] = None
        results["inference_time_error"] = "Failed to measure inference time"
    
    # ========== 5. Logging ==========
    adapter_status = "with adapter" if use_adapter else "without adapter"
    logger.info("=" * 80)
    logger.info(f"Model Complexity Metrics ({adapter_status.upper()})")
    logger.info("=" * 80)
    
    # Parameters
    logger.info(f"\n[Parameters]")
    logger.info(f"  Total parameters:      {total_params/1e6:.3f} M ({total_params/1e9:.6f} B)")
    logger.info(f"  Trainable parameters:  {trainable_params/1e6:.3f} M ({trainable_params/1e9:.6f} B)")
    logger.info(f"  Non-trainable:         {(total_params-trainable_params)/1e6:.3f} M")
    
    # Model Size
    logger.info(f"\n[Model Size]")
    if total_size_gb >= 1.0:
        logger.info(f"  Model size:           {total_size_gb:.3f} GB ({total_size_mb:.2f} MB)")
    else:
        logger.info(f"  Model size:           {total_size_mb:.2f} MB")
    
    # FLOPs/MACs
    logger.info(f"\n[Computational Complexity]")
    if total_macs is not None:
        if total_macs >= 1e9:
            logger.info(f"  Total MACs:           {total_macs/1e9:.3f} GMACs ({total_macs/1e6:.3f} MMACs)")
        elif total_macs >= 1e6:
            logger.info(f"  Total MACs:           {total_macs/1e6:.3f} MMACs ({total_macs/1e3:.3f} kMACs)")
        else:
            logger.info(f"  Total MACs:           {total_macs/1e3:.3f} kMACs")
        logger.info(f"  Total FLOPs:          {total_flops/1e9:.3f} GFLOPs")
    else:
        logger.info(f"  FLOPs/MACs:           Not available (install fvcore)")
    
    # Inference Time
    logger.info(f"\n[Inference Time]")
    if results.get("inference_time_ms") is not None:
        logger.info(f"  Average:              {avg_time_ms:.2f} ms/point cloud")
        logger.info(f"  Min:                  {min_time_ms:.2f} ms")
        logger.info(f"  Max:                  {max_time_ms:.2f} ms")
        if len(times) > 1:
            logger.info(f"  Std:                  {results['inference_time_std_ms']:.2f} ms")
    else:
        logger.info(f"  Inference time:       Not available")
    
    logger.info("=" * 80)
    logger.info("")
    
    return results


def log_model_complexity_comparison(
    model_without_adapter: nn.Module,
    model_with_adapter: nn.Module,
    tokenizer,
    point_cloud_shape: Tuple[int, int] = (1, 8192, 3),
    device: str = None,
    logger=None
):
    """
    Compare and log model complexity metrics for models with and without adapter.
    
    Args:
        model_without_adapter: Model instance with adapter disabled/removed
        model_with_adapter: Model instance with adapter enabled
        tokenizer: Tokenizer for creating input_ids
        point_cloud_shape: Shape of point cloud
        device: Device to run on
        logger: Logger instance
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPLEXITY COMPARISON")
    logger.info("=" * 80)
    
    # Measure without adapter
    logger.info("\n[Measuring model WITHOUT adapter...]")
    metrics_without = measure_model_complexity(
        model_without_adapter, tokenizer, point_cloud_shape, device=device, 
        use_adapter=False, logger=logger
    )
    
    # Measure with adapter
    logger.info("\n[Measuring model WITH adapter...]")
    metrics_with = measure_model_complexity(
        model_with_adapter, tokenizer, point_cloud_shape, device=device,
        use_adapter=True, logger=logger
    )
    
    # Comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\n{'Metric':<30} {'Without Adapter':<20} {'With Adapter':<20} {'Difference':<20}")
    logger.info("-" * 90)
    
    # Model size
    size_wo = metrics_without.get("model_size_MB", 0)
    size_w = metrics_with.get("model_size_MB", 0)
    diff_size = size_w - size_wo
    logger.info(f"{'Model Size (MB)':<30} {size_wo:<20.2f} {size_w:<20.2f} {diff_size:+.2f}")
    
    # Trainable parameters
    train_wo = metrics_without.get("trainable_parameters_M", 0)
    train_w = metrics_with.get("trainable_parameters_M", 0)
    diff_train = train_w - train_wo
    logger.info(f"{'Trainable Params (M)':<30} {train_wo:<20.3f} {train_w:<20.3f} {diff_train:+.3f}")
    
    # Total parameters
    total_wo = metrics_without.get("total_parameters_M", 0)
    total_w = metrics_with.get("total_parameters_M", 0)
    diff_total = total_w - total_wo
    logger.info(f"{'Total Params (M)':<30} {total_wo:<20.3f} {total_w:<20.3f} {diff_total:+.3f}")
    
    # MACs
    if metrics_without.get("total_kMACs") and metrics_with.get("total_kMACs"):
        macs_wo = metrics_without.get("total_kMACs", 0)
        macs_w = metrics_with.get("total_kMACs", 0)
        diff_macs = macs_w - macs_wo
        logger.info(f"{'kMACs':<30} {macs_wo:<20.3f} {macs_w:<20.3f} {diff_macs:+.3f}")
    
    # Inference time
    if metrics_without.get("inference_time_ms") and metrics_with.get("inference_time_ms"):
        time_wo = metrics_without.get("inference_time_ms", 0)
        time_w = metrics_with.get("inference_time_ms", 0)
        diff_time = time_w - time_wo
        logger.info(f"{'Inference Time (ms)':<30} {time_wo:<20.2f} {time_w:<20.2f} {diff_time:+.2f}")
    
    logger.info("=" * 80)
    logger.info("")
    
    return metrics_without, metrics_with


def log_model_complexity_simple(
    model: nn.Module,
    tokenizer,
    device: str = None,
    logger=None
):
    """
    Simple wrapper to measure and log model complexity.
    Automatically detects if adapter is enabled.
    
    Args:
        model: PointLLM model instance
        tokenizer: Tokenizer instance
        device: Device (auto-detect if None)
        logger: Logger instance (uses print if None)
    
    Returns:
        Dictionary with metrics
    """
    import logging
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # Check if adapter is enabled
    has_adapter = False
    if hasattr(model, 'get_model'):
        point_model = model.get_model()
        if hasattr(point_model, 'point_backbone'):
            neck = getattr(point_model.point_backbone, 'transform_neck3d', None)
            has_adapter = neck is not None
    
    # Determine point cloud shape
    point_cloud_shape = (1, 8192, 3)  # default
    if hasattr(model, 'get_model'):
        point_model = model.get_model()
        if hasattr(point_model, 'point_backbone_config'):
            config = point_model.point_backbone_config
            # Try to infer from config
            if 'point_token_len' in config:
                # Rough estimate: each token corresponds to ~128 points
                num_points = config.get('point_token_len', 64) * 128
                point_dims = config.get('point_cloud_dim', 3)
                point_cloud_shape = (1, num_points, point_dims)
    
    return measure_model_complexity(
        model=model,
        tokenizer=tokenizer,
        point_cloud_shape=point_cloud_shape,
        device=device,
        use_adapter=has_adapter,
        logger=logger
    )
