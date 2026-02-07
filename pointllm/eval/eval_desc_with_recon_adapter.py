# -*- coding: utf-8 -*-
"""
Eval PointLLM description on ModelNet40, with optional GRASP recon pre-processing.

Goal:
- PointLLM still outputs DESCRIPTION (like your eval_modelnet_cls.py)
- BUT you can optionally run point clouds through GRASP recon + recon_rate adapters first
  to quickly detect if recon training broke geometry.

Supports:
- load base PointLLM (RunsenXu_graspnet_enc_dec_r03_bak/PointLLM_7B_v1.2)
- optional recon: load GRASP ckpt + yaml + adapter_model.bin, run recon to get xyz_hat
- feed (xyz_hat or original xyz) into PointLLM for text generation
- print sample outputs for early debugging
"""

import os
import sys
import json
import yaml
import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import MinkowskiEngine as ME
from transformers import AutoTokenizer

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from types import SimpleNamespace


try:
    from pointllm.data import ModelNet
except Exception:
    ModelNet = None

from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat

# timing & stats (keep same style)
from pointllm.eval.timing_utils import attach_timers, DeviceTimer
from pointllm.eval.model_stats import (
    params_by_module, group_sum,
    flops_by_module, pretty_print_flops,
)

# ---------------- logging ----------------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("eval_desc")


PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.insert(0, os.path.join(project_root, "pointllm"))
from pccai.models.architectures.grasp import GeoResCompression  # noqa


# ---------------- small utils ----------------

def _extract_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return ckpt
    for k in ["model", "state_dict", "net", "params", "net_state_dict"]:
        if k in ckpt:
            return ckpt[k]
    return ckpt


def _get_modules_cfg(grasp_yaml: dict) -> dict:
    if isinstance(grasp_yaml, dict) and "modules" in grasp_yaml and isinstance(grasp_yaml["modules"], dict):
        return grasp_yaml["modules"]
    if isinstance(grasp_yaml, dict) and "net_config" in grasp_yaml and isinstance(grasp_yaml["net_config"], dict):
        return grasp_yaml["net_config"]
    return grasp_yaml


def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
    # [B,N,>=3] -> [B,N,3]
    xyz = point_clouds[:, :, :3].float()
    centroid = torch.mean(xyz, dim=1, keepdim=True)
    xyz = xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
    return xyz / (dist.unsqueeze(-1) + 1e-6)


def _sample_points(x: torch.Tensor, n: int) -> torch.Tensor:
    B, N, C = x.shape
    if N == n:
        return x
    if N > n:
        idx = torch.randint(0, N, (B, n), device=x.device)
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    pad = x.new_zeros((B, n - N, C))
    return torch.cat([x, pad], dim=1)


def chamfer_l2_approx(x: torch.Tensor, y: torch.Tensor, sample_n: int = 2048) -> torch.Tensor:
    x_s = _sample_points(x, sample_n)
    y_s = _sample_points(y, sample_n)
    d = torch.cdist(x_s, y_s, p=2)
    min_xy = d.min(dim=2)[0]
    min_yx = d.min(dim=1)[0]
    return (min_xy.mean(dim=1) + min_yx.mean(dim=1)).mean()


def _rate_from_likelihoods(lk):
    eps = 1e-9
    if lk is None:
        return None
    if torch.is_tensor(lk):
        return (-torch.log2(lk.clamp_min(eps))).mean()
    if isinstance(lk, dict):
        vals = []
        for v in lk.values():
            if torch.is_tensor(v):
                vals.append((-torch.log2(v.clamp_min(eps))).mean())
        return torch.stack(vals).mean() if len(vals) > 0 else None
    if isinstance(lk, (list, tuple)):
        vals = []
        for v in lk:
            if torch.is_tensor(v):
                vals.append((-torch.log2(v.clamp_min(eps))).mean())
            elif isinstance(v, dict):
                vv = _rate_from_likelihoods(v)
                if vv is not None:
                    vals.append(vv)
        return torch.stack(vals).mean() if len(vals) > 0 else None
    return None


# ---------------- PointLLM init ----------------

def init_pointllm_for_desc(args):
    disable_torch_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "fp16":
        llm_dtype = torch.float16
    elif args.dtype == "bf16":
        llm_dtype = torch.bfloat16
    else:
        llm_dtype = torch.float32

    logger.info(f"device={device}, dtype={llm_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    # ✅ Avoid AutoConfig KeyError('pointllm'): just load model directly
    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=llm_dtype if device != "cpu" else torch.float32,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to(device)

    # Ensure point tokens exist (some repos require this)
    try:
        tokenizer.add_tokens(["<point_patch>", "<point_start>", "<point_end>"], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass

    # Initialize point_backbone_config used in prompt formatting
    if hasattr(model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv = conv_templates["vicuna_v1_1"].copy()
    return model, tokenizer, conv


# ---------------- GRASP recon init ----------------

class SparseTensorAdapterWrap(nn.Module):
    """
    Wrap grasp_core.vox_enc:
      y = vox_enc(x)
      y = enc_adapter(y)  (MinkowskiAdapter on SparseTensor)
    Lazy init using y.F channels.
    """
    def __init__(self, module: nn.Module, ratio: int, enabled: bool, device: torch.device):
        super().__init__()
        self.module = module
        self.ratio = int(ratio)
        self.enabled = bool(enabled)
        self.device = device
        self.adapter = None
        self._inited = False

    def _ensure(self, y):
        if (not self.enabled) or self._inited:
            return
        if hasattr(y, "F") and torch.is_tensor(y.F):
            ch = int(y.F.shape[1])
            from pointllm.pccai.models.modules.adapters import MinkowskiAdapter
            self.adapter = MinkowskiAdapter(
                channels=ch,
                ratio=self.ratio,
                enabled=True,
                use_alpha=True,
            ).to(self.device).float()
            self._inited = True
            logger.info(f"✅ [GRASP ENC] init MinkowskiAdapter(channels={ch}, ratio={self.ratio})")

    def forward(self, *args, **kwargs):
        y = self.module(*args, **kwargs)
        if self.enabled and hasattr(y, "F") and torch.is_tensor(y.F):
            if not self._inited:
                self._ensure(y)
            if self.adapter is not None:
                y = self.adapter(y)
        return y


class GraspReconEngine(nn.Module):
    """
    Given point_clouds [B,N,3], returns:
      xyz_hat [B,N,3], rate_loss (scalar tensor or None), out_dict (raw grasp RunsenXu_graspnet_enc_dec_r04)
    """
    def __init__(self, grasp_core: nn.Module, voxel_size: float, npoints: int):
        super().__init__()
        self.core = grasp_core
        self.voxel_size = float(voxel_size)
        self.npoints = int(npoints)

    @staticmethod
    def _points_to_me_coords(xyz: torch.Tensor, voxel_size_: float) -> torch.Tensor:
        # xyz: [B,N,3] float normalized
        B, N, _ = xyz.shape
        coords_list = []
        for b in range(B):
            q = torch.floor(xyz[b, :, :3] / voxel_size_).to(torch.int32)
            try:
                uq = ME.utils.sparse_quantize(q)
            except Exception:
                uq = torch.unique(q, dim=0)
            coords_list.append(uq)
        coords = ME.utils.batched_coordinates(coords_list, dtype=torch.int32)
        return coords

    @staticmethod
    def _coords_to_xyz_sequence(x_hat: torch.Tensor, B: int, N_: int, voxel_size_: float, device_: torch.device) -> torch.Tensor:
        # x_hat: [M',4] int (b,x,y,z)
        b_idx = x_hat[:, 0].long()
        xyz = x_hat[:, 1:4].float() * voxel_size_
        seq = []
        for b in range(B):
            pts_b = xyz[b_idx == b]
            if pts_b.numel() == 0:
                pts_b = xyz.new_zeros((0, 3))
            if pts_b.shape[0] > N_:
                idx = torch.randint(0, pts_b.shape[0], (N_,), device=device_)
                pts_b = pts_b[idx]
            elif pts_b.shape[0] < N_:
                pad = pts_b.new_zeros((N_ - pts_b.shape[0], 3))
                pts_b = torch.cat([pts_b, pad], dim=0)
            seq.append(pts_b.unsqueeze(0))
        return torch.cat(seq, dim=0)

    def forward(self, point_clouds: torch.Tensor):
        xyz_norm = _normalize_unit_sphere(point_clouds)
        with torch.cuda.amp.autocast(enabled=False):
            coords_int = self._points_to_me_coords(xyz_norm, self.voxel_size).to(point_clouds.device).int()
            out = self.core.forward(coords_int)

            if not isinstance(out, dict) or "x_hat" not in out:
                raise RuntimeError("GRASP RunsenXu_graspnet_enc_dec_r04 missing 'x_hat'.")

            lk = out.get("likelihoods", None) if isinstance(out, dict) else None
            rate = _rate_from_likelihoods(lk)

            x_hat = out["x_hat"]
            B = int(point_clouds.shape[0])
            xyz_hat = self._coords_to_xyz_sequence(x_hat, B=B, N_=self.npoints, voxel_size_=self.voxel_size, device_=point_clouds.device)

        return xyz_hat, rate, out

def _remap_adapter_keys_for_grasp(sd: dict) -> dict:
    """
    adapter_model.bin saved from the whole PointLLM model has prefixes like:
      model.point_backbone.core.vox_dec.adapter1....
    but GeoResCompression expects:
      vox_dec.adapter1....
    """
    out = {}
    for k, v in sd.items():
        k2 = k

        # most common prefix from your logs
        if k2.startswith("model.point_backbone.core."):
            k2 = k2[len("model.point_backbone.core."):]
        # other possible variants (keep them; harmless if not matched)
        if k2.startswith("point_backbone.core."):
            k2 = k2[len("point_backbone.core."):]
        if k2.startswith("core."):
            k2 = k2[len("core."):]

        out[k2] = v
    return out

def init_grasp_recon_engine(args, device: str):
    from pointllm.pccai.models.architectures.grasp import GeoResCompression
    from types import SimpleNamespace
    import yaml, torch
    import MinkowskiEngine as ME

    grasp_yaml = yaml.safe_load(open(args.grasp_config, "r"))
    modules_cfg = _get_modules_cfg(grasp_yaml)

    core = GeoResCompression(modules_cfg, SimpleNamespace(phase="test")).to(device).float()
    core.load_state_dict(_extract_state_dict(args.grasp_ckpt), strict=False)
    core = core.eval().float()

    # 1) wrap encoder (lazy)
    if args.use_enc_adapter and hasattr(core, "vox_enc") and core.vox_enc is not None:
        core.vox_enc = SparseTensorAdapterWrap(
            core.vox_enc,
            ratio=args.enc_adapter_ratio,
            enabled=True,
            device=torch.device(device),
        ).to(device).float()
        logger.info("✅ Wrapped core.vox_enc for encoder adapter (lazy).")

    # 2) warmup BEFORE loading adapter weights -> create lazy params
    try:
        with torch.no_grad():
            # minimal dummy coords to trigger vox_enc forward
            # coords format: [M,4] int32 (b,x,y,z)
            dummy = torch.tensor(
                [[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                device=device, dtype=torch.int32
            )
            _ = core.forward(dummy)
        logger.info("✅ GRASP core warmup done (lazy enc adapter should be instantiated).")
    except Exception as e:
        logger.warning(f"⚠️ GRASP warmup (pre-load) failed: {e}")

    # 3) load adapter bin after warmup
    sd_raw = torch.load(args.adapter_bin_path, map_location="cpu")
    sd = _remap_adapter_keys_for_grasp(sd_raw)

    # ✅ optional: filter to keys that exist (avoid unused garbage)
    core_keys = set(core.state_dict().keys())
    sd = {k:v for k,v in sd.items() if k in core_keys}

    missing, unexpected = core.load_state_dict(sd, strict=False)
    matched = len(sd) - len(unexpected)

    logger.info("=" * 80)
    logger.info(f"✅ Loaded adapter_model.bin: {args.adapter_bin_path}")
    logger.info(f"  tensors(after filter): {len(sd)}")
    logger.info(f"  matched tensors loaded : {matched}")
    logger.info(f"  unexpected (not used)  : {len(unexpected)}")
    logger.info(f"  missing (core extra)   : {len(missing)}  (normal)")
    if len(unexpected) > 0:
        logger.info("  Sample unexpected keys:")
        for k in list(unexpected)[:20]:
            logger.info(f"    - {k}")
    logger.info("=" * 80)

    engine = GraspReconEngine(core, voxel_size=args.voxel_size, npoints=args.npoints).to(device).float()
    return engine, matched, len(sd_raw)



# ---------------- dataset / loader ----------------

def load_dataset(args):
    logger.info(f"Loading {args.split} split of ModelNet dataset...")
    if args.use_dir or ModelNet is None:
        ds = ModelNet40DirCompat(
            root=args.modelnet_root,
            split=args.split,
            npoints=args.npoints,
            cache_npy=True,
            subset_nums=args.subset_nums
        )
    else:
        ds = ModelNet(
            config_path=args.config_path if hasattr(args, "config_path") else None,
            split=args.split,
            subset_nums=args.subset_nums,
            use_color=args.use_color
        )
    logger.info("Done.")
    return ds


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=8):
    assert shuffle is False
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ---------------- generation ----------------

def build_prompt(conv, model, tokenizer, prompt_index: int):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    # pbc 可能存在，也可能不存在某些 key（Internal 分支经常不全）
    pbc = getattr(model.get_model(), "point_backbone_config", {}) or {}

    # ✅ point_token_len: 如果没有就给个常用默认值 64（你也可改成 args 传入）
    point_token_len = int(pbc.get("point_token_len", 64))

    # ✅ token 字符串：优先用 pbc 里有的，否则用 tokenizer 里你 add_tokens 的名字
    patch_tok = pbc.get("default_point_patch_token", "<point_patch>")
    start_tok = pbc.get("default_point_start_token", "<point_start>")
    end_tok   = pbc.get("default_point_end_token", "<point_end>")

    # ✅ start/end 是否启用：没有就默认为 False（和你旧脚本一致）
    mm_use_point_start_end = bool(pbc.get("mm_use_point_start_end", False))

    # ---- 确保 tokenizer 确实认识这些 token（否则拼进去也没意义）----
    # 这里不是 try/except “防御”，而是保证 prompt token 能被正确编码
    for t in [patch_tok, start_tok, end_tok]:
        if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id:
            # 如果 pbc 给了奇怪 token 名，回退到我们标准 token
            if t == patch_tok:
                patch_tok = "<point_patch>"
            elif t == start_tok:
                start_tok = "<point_start>"
            elif t == end_tok:
                end_tok = "<point_end>"

    if mm_use_point_start_end:
        qs = start_tok + (patch_tok * point_token_len) + end_tok + "\n" + qs
    else:
        qs = (patch_tok * point_token_len) + "\n" + qs

    conv2 = conv.copy()
    conv2.append_message(conv2.roles[0], qs)
    conv2.append_message(conv2.roles[1], None)
    prompt = conv2.get_prompt()
    return prompt, stop_str



def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria,
                     do_sample=True, temperature=1.0, top_k=50, top_p=0.95, max_length=2048):
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            stopping_criteria=[stopping_criteria]
        )
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    return [o.strip() for o in outputs]


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()

    # base model for text generation
    parser.add_argument("--base_model_path", type=str, required=True)

    # dataset
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--use_dir", action="store_true", help="force dir-compat dataset")
    parser.add_argument("--modelnet_root", type=str, required=True)
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--subset_nums", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--use_color", action="store_true", default=True)

    # generation
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    # RunsenXu_graspnet_enc_dec_r04
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    # sample printing (debug)
    parser.add_argument("--print_samples", action="store_true", default=True)
    parser.add_argument("--sample_every", type=int, default=50, help="print samples every N batches")
    parser.add_argument("--num_sample_print", type=int, default=2, help="how many samples to print each time")

    # recon pre-processing (optional)
    parser.add_argument("--use_recon", action="store_true", default=False,
                        help="if set: point clouds -> GRASP recon (with adapter_model.bin) -> feed recon to PointLLM")
    parser.add_argument("--adapter_bin_path", type=str, default=None)
    parser.add_argument("--grasp_ckpt", type=str, default=None)
    parser.add_argument("--grasp_config", type=str, default=None)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--enc_adapter_ratio", type=int, default=8)
    parser.add_argument("--use_enc_adapter", action="store_true", default=True)
    parser.add_argument("--compute_recon_metrics", action="store_true", default=True,
                        help="if set: compute chamfer+rate for recon samples (costs some time)")
    parser.add_argument("--chamfer_sample_n", type=int, default=2048)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    # RunsenXu_graspnet_enc_dec_r04 paths
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_model_path.replace("/", "_"), "evaluation_recon" if args.use_recon else "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file is None:
        args.output_file = f"ModelNet_desc_prompt{args.prompt_index}{'_recon' if args.use_recon else ''}.json"
    out_path = os.path.join(args.output_dir, args.output_file)
    logger.info(f"RunsenXu_graspnet_enc_dec_r04: {out_path}")

    # dataset
    ds = load_dataset(args)
    dl = get_dataloader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # init base pointllm
    model, tokenizer, conv = init_pointllm_for_desc(args)
    timers, _hooks = attach_timers(model, tokenizer)
    logger.info("[Timing] timers attached (tokenizer / point_encoder / projector).")

    # init recon engine if needed
    recon_engine = None
    if args.use_recon:
        assert args.adapter_bin_path and args.grasp_ckpt and args.grasp_config, \
            "--use_recon requires --adapter_bin_path --grasp_ckpt --grasp_config"
        recon_engine, matched, total = init_grasp_recon_engine(args, device=device)
        if matched == 0:
            logger.warning("⚠️ recon adapter matched=0. This usually means your adapter keys do not align with GRASP core modules.")
        # warm up once to init lazy enc adapter (vox_enc wrap)
        try:
            recon_engine.eval()
            with torch.no_grad():
                b0 = next(iter(dl))
                _ = recon_engine(b0["point_clouds"][:1].to(device).float())
            logger.info("✅ Recon engine warmup ok.")
        except Exception as e:
            logger.warning(f"Recon engine warmup failed: {e}")

    # params summary (like your script)
    core = model.get_model()
    rows, total_params = params_by_module(core)
    prefixes = {
        "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
        "Projector": ["point_proj", "mm_projector", "projector"],
        "LLM": ["language_model", "model", "transformer", "lm"],
    }
    grouped_params = group_sum({name: n for name, _, n in rows}, prefixes)
    logger.info("Grouped Params (M): " + str({k: v / 1e6 for k, v in grouped_params.items()}))

    # build prompt once
    prompt, stop_str = build_prompt(conv, model, tokenizer, args.prompt_index)

    tok_timer = timers.get("tokenizer", DeviceTimer("tokenizer"))
    with tok_timer:
        tok = tokenizer([prompt])
    timers["tokenizer"] = tok_timer
    input_ids_ = torch.as_tensor(tok.input_ids).to(model.device)

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    # FLOPs on first batch (optional; keep like your style)
    try:
        it0 = iter(dl)
        first_batch = next(it0)
        pc1 = first_batch["point_clouds"][:1].to(model.device).to(model.dtype)
        ids1 = input_ids_[:1]
        by_mod, flops_total = flops_by_module(core.eval(), ids1, pc1)
        pretty_print_flops(by_mod, flops_total, topk=99999)
        grouped = group_sum(by_mod, prefixes)
        logger.info("Grouped FLOPs (GFLOPs, single forward): " + str({k: v / 1e9 for k, v in grouped.items()}))
    except Exception as e:
        logger.warning(f"FLOPs analysis failed: {e}")

    # iterate + generate
    results = {"prompt": PROMPT_LISTS[args.prompt_index], "use_recon": bool(args.use_recon), "results": []}

    # IMPORTANT: use a fresh iterator because we consumed one batch in FLOPs
    dl_iter = iter(dl)
    pbar = tqdm(dl_iter, total=len(dl), desc="Generating")

    for step, batch in enumerate(pbar):
        pc = batch["point_clouds"].to(device).float()  # always float for recon
        labels = batch.get("labels", None)
        label_names = batch.get("label_names", None)
        indice = batch.get("indice", None)

        # optional recon
        chamfer_val = None
        rate_val = None
        pc_used = pc

        if recon_engine is not None:
            recon_engine.eval()
            with torch.no_grad():
                xyz_hat, rate, _out = recon_engine(pc)
            # normalize recon to unit sphere before feeding to PointLLM (safer)
            pc_used = _normalize_unit_sphere(xyz_hat).to(device).to(model.dtype)

            if args.compute_recon_metrics:
                with torch.no_grad():
                    gt = _normalize_unit_sphere(pc)
                    ch = chamfer_l2_approx(_normalize_unit_sphere(xyz_hat), gt, sample_n=args.chamfer_sample_n)
                    chamfer_val = float(ch.detach().cpu().item())
                    if rate is not None:
                        rate_val = float(rate.detach().cpu().item())

        else:
            # feed original point cloud
            pc_used = pc.to(device).to(model.dtype)

        bs = pc_used.shape[0]
        input_ids = input_ids_.repeat(bs, 1)

        outs = generate_outputs(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            point_clouds=pc_used,
            stopping_criteria=stopping_criteria,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_length=args.max_length,
        )

        # sample prints (early debug)
        if args.print_samples and (step % max(1, args.sample_every) == 0):
            k = min(args.num_sample_print, bs)
            logger.info("=" * 80)
            logger.info(f"[SAMPLE] step={step}, use_recon={args.use_recon}, chamfer={chamfer_val}, rate={rate_val}")
            for i in range(k):
                oid = int(indice[i].item()) if indice is not None else (step * bs + i)
                gtname = label_names[i] if label_names is not None else "N/A"
                logger.info(f"  object_id={oid}, gt={gtname}")
                logger.info(f"  RunsenXu_graspnet_enc_dec_r04: {outs[i]}")
            logger.info("=" * 80)

        # save rows
        for i in range(bs):
            rec = {
                "object_id": int(indice[i].item()) if indice is not None else (step * bs + i),
                "ground_truth": int(labels[i].item()) if labels is not None else -1,
                "label_name": label_names[i] if label_names is not None else None,
                "model_output": outs[i],
            }
            if chamfer_val is not None:
                rec["recon_chamfer_l2_approx"] = chamfer_val
            if rate_val is not None:
                rec["recon_rate"] = rate_val
            results["results"].append(rec)

    # dump json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_path}")

    # timing summary
    if timers is not None:
        def _fmt(t):
            s = t.stats()
            return f"{s['mean_ms']:.3f} ± {s['std_ms']:.3f} ms (n={s['n']})"

        print("\n===== Inference Component Timing =====")
        if "tokenizer" in timers:
            print("Tokenizer        :", _fmt(timers["tokenizer"]))
        for k, v in timers.items():
            if k == "tokenizer":
                continue
            print(f"{k:16s}: {_fmt(v)}")
        print("======================================\n")


if __name__ == "__main__":
    main()
