# -*- coding: utf-8 -*-
"""
Enhanced Eval PointLLM (Internal) on ModelNet40 — train-aligned injection + logprob classification scoring

Aligned to your CE-training train_internal_adapter.py:
- tokenizer special tokens: ONLY add <point_patch>, <point_start>, <point_end> (do NOT add <point>)
- dec_adapter init uses alpha_init=-4.0 (same as training)
- point_backbone_config set on BOTH model.config and model.get_model().config
- AMP dtype follows args (bf16/fp16/fp32)

Sanities:
1) CKPT key coverage report
2) Point influence sanity (logits diff)
3) A/B sensitivity hint
"""

import os
import sys
import glob
import json
import yaml
import argparse
import logging
from types import SimpleNamespace
from typing import Dict, Tuple, Optional, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.conversation import conv_templates, SeparatorStyle

# --- path inject ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pointllm.model.pointllm_internal import PointLLMLlamaForCausalLMInternal
from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat

# IMPORTANT: use pccai import style to match your environment
sys.path.insert(0, os.path.join(project_root, "pointllm"))
from pccai.models.architectures.grasp import GeoResCompression  # noqa
from pccai.models.modules.adapters import MinkowskiAdapter, ResidualConv1dAdapter  # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvalInternalEnhancedAligned")

DEBUG = int(os.environ.get("POINTLLM_DEBUG", "0"))
LOCAL_FILES_ONLY = int(os.environ.get("LOCAL_FILES_ONLY", "1"))

DEFAULT_POINT_PATCH_TOKEN = "<point_patch>"
POINT_TOKEN = "<point>"          # NOTE: placeholder in prompt, NOT a tokenizer special token
POINT_START = "<point_start>"
POINT_END = "<point_end>"

MODELNET40_CATEGORIES = [
    "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair","cone",
    "cup","curtain","desk","door","dresser","flower_pot","glass_box","guitar","keyboard","lamp",
    "laptop","mantel","monitor","night_stand","person","piano","plant","radio","range_hood","sink",
    "sofa","stairs","stool","table","tent","toilet","tv_stand","vase","wardrobe","xbox"
]

PROMPT_LISTS = [
    "What is this object? Answer with one word from the ModelNet40 category list.",
    "The object category is: ",
]


# ------------------------- utils -------------------------
def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _pick_latest_bin(ckpt_dir: str) -> str:
    """
    Pick the correct adapter weight file from a HF Trainer checkpoint dir.

    Prefer:
      1) adapter_model.bin (PEFT-style)
      2) pytorch_model.bin (full model)
      3) model*.bin (fallback)
    Explicitly exclude non-weight bins (training args / optimizer / scheduler / trainer state).
    """
    # 1) explicit best choice
    cand = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.exists(cand):
        return cand

    # 2) common HF weight filename
    cand = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(cand):
        return cand

    # 3) scan *.bin but exclude known non-weight artifacts
    exclude_names = {
        "training_args.bin",
        "trainer_state.bin",
        "optimizer.bin",
        "scheduler.bin",
        "scaler.bin",
    }

    all_bins = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.bin")):
        base = os.path.basename(p)
        if base in exclude_names:
            continue
        # extra safety: exclude things that clearly aren't weights
        low = base.lower()
        if "training_args" in low or "trainer_state" in low or "optimizer" in low or "scheduler" in low:
            continue
        all_bins.append(p)

    if not all_bins:
        raise FileNotFoundError(
            f"❌ No model weight .bin found under: {ckpt_dir}. "
            f"Expected adapter_model.bin / pytorch_model.bin."
        )

    # pick latest among remaining candidates
    all_bins = sorted(all_bins, key=os.path.getmtime, reverse=True)
    return all_bins[0]


def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
    xyz = point_clouds[:, :, :3].float()
    centroid = torch.mean(xyz, dim=1, keepdim=True)
    xyz = xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
    return xyz / (dist.unsqueeze(-1) + 1e-6)

def _build_prompt(conv_mode: str, question: str):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], f"{POINT_TOKEN}\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return prompt, stop_str

def _replace_point_token(prompt: str, point_token_len: int):
    patch_seq = " ".join([DEFAULT_POINT_PATCH_TOKEN] * int(point_token_len))
    rep = f"{POINT_START}\n{patch_seq}\n{POINT_END}"
    return prompt.replace(POINT_TOKEN, rep)

def _count_tokens(tokenizer, prompt: str):
    ids = tokenizer([prompt], return_tensors="pt").input_ids[0]
    def _id(tok):
        return int(tokenizer.convert_tokens_to_ids(tok))
    # For placeholder <point>, it likely becomes unknown or split tokens. That's OK.
    return {
        "<point>": int((ids == _id(POINT_TOKEN)).sum().item()) if _id(POINT_TOKEN) != tokenizer.unk_token_id else 0,
        "<point_patch>": int((ids == _id(DEFAULT_POINT_PATCH_TOKEN)).sum().item()),
        "<point_start>": int((ids == _id(POINT_START)).sum().item()),
        "<point_end>": int((ids == _id(POINT_END)).sum().item()),
    }

def _extract_modules_cfg(cfg_any: dict) -> dict:
    if isinstance(cfg_any, dict):
        if "modules" in cfg_any and isinstance(cfg_any["modules"], dict):
            return cfg_any["modules"]
        if "net_config" in cfg_any and isinstance(cfg_any["net_config"], dict):
            return cfg_any["net_config"]
    return cfg_any

def _load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "net_state_dict" in obj and isinstance(obj["net_state_dict"], dict):
        return obj["net_state_dict"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"❌ {path} is not a dict-like checkpoint.")

def _prefix_rewrite(sd: Dict[str, torch.Tensor], rules: Tuple[Tuple[str, str], ...]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        kk = k
        for src, dst in rules:
            if kk.startswith(src):
                kk = dst + kk[len(src):]
        out[kk] = v
    return out

def safe_load_state_dict_by_shape(model: nn.Module, sd_in: Dict[str, torch.Tensor], tag: str = "") -> int:
    msd = model.state_dict()
    filtered = {}
    skipped = 0
    loaded = 0
    for k, v in sd_in.items():
        if k not in msd:
            continue
        if tuple(msd[k].shape) == tuple(v.shape):
            filtered[k] = v
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(filtered, strict=False)
    logger.info(f"[{tag}] loaded={loaded} skipped_shape_mismatch={skipped} (candidate={len(sd_in)})")
    return loaded

def _get_label_name(batch: dict) -> Optional[str]:
    if "label_name" in batch:
        ln = batch["label_name"]
        if isinstance(ln, (list, tuple)):
            return str(ln[0])
        return str(ln)
    for k in ["label", "labels", "y", "category", "cls"]:
        if k in batch:
            v = batch[k]
            if torch.is_tensor(v):
                idx = int(v[0].item())
            elif isinstance(v, (list, tuple)):
                idx = int(v[0])
            else:
                idx = int(v)
            if 0 <= idx < len(MODELNET40_CATEGORIES):
                return MODELNET40_CATEGORIES[idx]
            return str(idx)
    return None

def _print_feat_stats(hat_f_seq: torch.Tensor, prefix: str = "[GRASP]"):
    x = hat_f_seq.detach().float()
    logger.info(
        f"{prefix} hat_f_seq stats: mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}"
    )

def _ckpt_key_report(sd: Dict[str, torch.Tensor], title: str = "[CKPT_REPORT]"):
    keys = list(sd.keys())

    def _hits(substrs: List[str]):
        return [k for k in keys if any(s in k for s in substrs)]

    hit_point_proj = _hits(["point_proj"])
    hit_pre_proj = _hits(["pre_proj_adapter"])
    hit_backbone = _hits(["model.point_backbone."])
    hit_dec_alpha = _hits(["dec_adapter.alpha", "dec_adapter.alpha_param", "dec_adapter.alpha_logit"])

    logger.info(f"{title} total_keys={len(keys)}")
    logger.info(f"{title} hit model.point_backbone.* : {len(hit_backbone)}")
    logger.info(f"{title} hit point_proj*          : {len(hit_point_proj)}")
    logger.info(f"{title} hit pre_proj_adapter*    : {len(hit_pre_proj)}")
    logger.info(f"{title} hit dec_adapter.alpha*   : {len(hit_dec_alpha)}")

    for name, arr in [
        ("point_proj", hit_point_proj),
        ("pre_proj_adapter", hit_pre_proj),
        ("point_backbone", hit_backbone[:10]),
    ]:
        if not arr:
            continue
        logger.info(f"{title} sample keys for {name}:")
        for k in arr[:20]:
            logger.info(f"  {k}")

def _has_any_keys(sd: Dict[str, torch.Tensor], contains_any: List[str]) -> bool:
    for k in sd.keys():
        for s in contains_any:
            if s in k:
                return True
    return False


# ------------------------- model init -------------------------
def init_model_internal(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        use_fast=False,
        local_files_only=bool(LOCAL_FILES_ONLY),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ✅ TRAIN-ALIGNED: ONLY add these 3 tokens (do NOT add <point> as special token)
    tokenizer.add_tokens([DEFAULT_POINT_PATCH_TOKEN, POINT_START, POINT_END], special_tokens=True)

    # dtype selection
    if getattr(args, "fp32", False):
        llm_dtype = torch.float32
        amp_dtype = None
    elif getattr(args, "fp16", False):
        llm_dtype = torch.float16
        amp_dtype = torch.float16
    else:
        llm_dtype = torch.bfloat16  # default
        amp_dtype = torch.bfloat16

    model = PointLLMLlamaForCausalLMInternal.from_pretrained(
        args.base_model_path,
        torch_dtype=llm_dtype,
        low_cpu_mem_usage=False,
        local_files_only=bool(LOCAL_FILES_ONLY),
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    patch_id = int(tokenizer.convert_tokens_to_ids(DEFAULT_POINT_PATCH_TOKEN))
    if patch_id < 0 or patch_id == tokenizer.unk_token_id:
        raise RuntimeError("❌ <point_patch> token id invalid. tokenizer.add_tokens may have failed.")

    point_cfg = {
        "point_token_len": int(args.point_token_len),
        "backbone_output_dim": int(args.backbone_output_dim),
        "point_patch_token_id": patch_id,
    }
    # ✅ set on both outer + inner config (train-aligned)
    model.config.point_backbone_config = point_cfg
    try:
        model.get_model().point_backbone_config = point_cfg
        model.get_model().config.point_backbone_config = point_cfg
    except Exception:
        pass

    adapter_path = _pick_latest_bin(args.my_checkpoint_dir)

    # =========================
    # ✅ FIX: robust torch.load for "adapter_model.bin" that contains pickled objects
    # Root cause: checkpoint includes a pickled TrainingArguments with module path
    # "pointllm.eval.eval_internal.TrainingArguments".
    # We inject a compatible symbol before loading.
    # =========================
    import pickle

    # 先保证普通 unpickle 找得到 TrainingArguments（应对 weights_only=False 的回退）
    try:
        import transformers as _tf
        globals()["TrainingArguments"] = getattr(_tf, "TrainingArguments", object)
    except Exception:
        class TrainingArguments:  # fallback stub
            pass

        globals()["TrainingArguments"] = TrainingArguments

    # 先尝试 weights_only=True（更安全），失败就回退 weights_only=False（你自己的 ckpt 可接受）
    try:
        full_sd = torch.load(adapter_path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, Exception) as e:
        logger.warning(
            f"[CKPT_LOAD] weights_only=True failed ({type(e).__name__}: {e}). "
            f"Falling back to weights_only=False (ONLY safe for trusted ckpt)."
        )
        full_sd = torch.load(adapter_path, map_location="cpu", weights_only=False)

    # 一些 ckpt 会把真正的权重藏在 state_dict/net_state_dict 里，这里顺手兼容一下
    if isinstance(full_sd, dict):
        if "state_dict" in full_sd and isinstance(full_sd["state_dict"], dict):
            full_sd = full_sd["state_dict"]
        elif "net_state_dict" in full_sd and isinstance(full_sd["net_state_dict"], dict):
            full_sd = full_sd["net_state_dict"]

    logger.info(f"Using adapter weight: {adapter_path}")

    if args.ckpt_report:
        _ckpt_key_report(full_sd, title="[CKPT_REPORT]")

    # ---- AUTO-DISABLE adapters if ckpt doesn't contain their weights ----
    want_enc = bool(int(args.use_enc_adapter))
    want_dec = bool(int(args.use_dec_adapter))

    has_enc = _has_any_keys(full_sd, ["model.point_backbone.enc_adapter", "enc_adapter."])
    has_dec = _has_any_keys(full_sd, ["model.point_backbone.dec_adapter", "dec_adapter."])

    if want_enc and not has_enc:
        logger.warning("[AUTO_DISABLE] use_enc_adapter=1 but ckpt has NO enc_adapter weights. Forcing use_enc_adapter=0.")
        args.use_enc_adapter = 0
    if want_dec and not has_dec:
        logger.warning("[AUTO_DISABLE] use_dec_adapter=1 but ckpt has NO dec_adapter weights. Forcing use_dec_adapter=0.")
        args.use_dec_adapter = 0

    yaml_cfg_raw = _load_yaml(args.grasp_config)
    grasp_cfg_used = _extract_modules_cfg(yaml_cfg_raw)
    core = GeoResCompression(grasp_cfg_used, SimpleNamespace(phase="test")).to(device).float()

    C_out = int(args.backbone_output_dim)
    T_tokens = int(args.point_token_len)

    class BBackboneWrapper(nn.Module):
        """
        Train-aligned wrapper:
          - core: GeoResCompression
          - enc_adapter: MinkowskiAdapter (lazy)
          - dec_adapter: ResidualConv1dAdapter (sigmoid gate) with alpha_init=-4.0 (same as train)
          - adapter1 kept for ckpt compat
        """
        def __init__(self, core_: nn.Module):
            super().__init__()
            self.core = core_

            self.enc_adapter = None
            self._enc_ratio = int(args.enc_adapter_ratio)
            self._enc_enabled = bool(int(args.use_enc_adapter))

            self.dec_adapter = None
            if bool(int(args.use_dec_adapter)):
                self.dec_adapter = ResidualConv1dAdapter(
                    channels=C_out,
                    ratio=int(args.dec_adapter_ratio),
                    use_alpha=True,
                    alpha_init=-4.0,  # ✅ TRAIN-ALIGNED
                ).to(device).float()

            self.adapter1 = nn.Linear(C_out, 768).to(device).float()

        def _ensure_enc_adapter(self, in_dim: int):
            if self.enc_adapter is not None:
                return
            self.enc_adapter = MinkowskiAdapter(
                channels=in_dim,
                ratio=self._enc_ratio,
                enabled=True,
                use_alpha=True,
            ).to(device).float()
            logger.info(f"✅ enc_adapter init: MinkowskiAdapter(channels={in_dim}, ratio={self._enc_ratio})")

        @staticmethod
        def _pad_crop_tokens(x: torch.Tensor, T_: int):
            B_, t_, C_ = x.shape
            if t_ > T_:
                return x[:, :T_, :]
            if t_ < T_:
                pad = x.new_zeros((B_, T_ - t_, C_))
                return torch.cat([x, pad], dim=1)
            return x

        def forward(self, point_clouds: torch.Tensor):
            xyz_norm = _normalize_unit_sphere(point_clouds)
            dev = xyz_norm.device

            hat_f_seq, R_loss, D_loss = None, None, None

            used_encode_decode = False
            if hasattr(self.core, "encode") and hasattr(self.core, "decode"):
                try:
                    used_encode_decode = True
                    enc_out = self.core.encode(xyz_norm)
                    if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 2:
                        y_hat, likelihoods = enc_out[0], enc_out[1]
                    else:
                        y_hat, likelihoods = enc_out, None

                    if self._enc_enabled and hasattr(y_hat, "F"):
                        self._ensure_enc_adapter(int(y_hat.F.shape[1]))
                        y_hat = self.enc_adapter(y_hat)

                    dec_out = self.core.decode(y_hat)
                    if isinstance(dec_out, (tuple, list)) and len(dec_out) >= 1:
                        hat_f_seq = dec_out[0]
                        D_loss = dec_out[1] if len(dec_out) > 1 and torch.is_tensor(dec_out[1]) else None
                    else:
                        hat_f_seq = dec_out

                    eps = 1e-9
                    if torch.is_tensor(likelihoods):
                        R_loss = (-torch.log2(likelihoods.clamp_min(eps))).mean()
                    elif isinstance(likelihoods, dict):
                        lk = likelihoods.get("feats", None)
                        R_loss = (-torch.log2(lk.clamp_min(eps))).mean() if torch.is_tensor(lk) else None

                except Exception as e:
                    used_encode_decode = False
                    if DEBUG:
                        logger.warning(f"⚠️ encode/decode failed, fallback forward(return_loss=True). err={e}")

            if not used_encode_decode:
                try:
                    out = self.core.forward(xyz_norm, return_loss=True)
                except TypeError:
                    out = self.core.forward(xyz_norm)

                if isinstance(out, (tuple, list)) and len(out) == 3 and torch.is_tensor(out[0]) and out[0].ndim == 3:
                    hat_f_seq, R_loss, D_loss = out
                else:
                    raise RuntimeError("❌ GRASP forward did not return (hat_f_seq, R_loss, D_loss).")

            if not torch.is_tensor(hat_f_seq):
                raise RuntimeError("❌ hat_f_seq is not a tensor.")
            if int(hat_f_seq.shape[-1]) != C_out:
                raise RuntimeError(f"❌ GRASP output dim mismatch: got C={int(hat_f_seq.shape[-1])}, expected {C_out}.")

            hat_f_seq = self._pad_crop_tokens(hat_f_seq, T_tokens)

            if self.dec_adapter is not None:
                hat_f_seq = self.dec_adapter(hat_f_seq)

            if args.print_grasp_stats:
                _print_feat_stats(hat_f_seq)

            if args.zero_point_ab:
                hat_f_seq = torch.zeros_like(hat_f_seq)

            if R_loss is None:
                R_loss = torch.zeros((), device=dev)
            if D_loss is None:
                D_loss = torch.zeros((), device=dev)

            return hat_f_seq, R_loss, D_loss, torch.zeros((), device=dev)

    wrapper = BBackboneWrapper(core).to(device).float()

    # ✅ Load order:
    # (A) r03 baseline -> wrapper.core
    grasp_sd_raw = _load_state_dict_any(args.grasp_ckpt)
    grasp_sd_raw = _prefix_rewrite(grasp_sd_raw, (("pcc_model.", "core."), ("module.", "core.")))
    _ = safe_load_state_dict_by_shape(wrapper, grasp_sd_raw, tag="POINT_BACKBONE_CORE_FROM_R03")

    # (B) adapter subtree override -> wrapper
    sd_wrapper = {}
    prefix = "model.point_backbone."
    for k, v in full_sd.items():
        if k.startswith(prefix):
            sd_wrapper[k[len(prefix):]] = v
    loaded_w = safe_load_state_dict_by_shape(wrapper, sd_wrapper, tag="POINT_BACKBONE_FROM_ADAPTER_BIN")
    logger.info(f"✅ wrapper load summary: from_adapter_bin={loaded_w}")

    # mount wrapper
    model.get_model().point_backbone = wrapper

    # (C) whole-model adapter load: point_proj / pre_proj_adapter / etc
    _ = safe_load_state_dict_by_shape(model, full_sd, tag="ADAPTER_BIN_TO_WHOLE_MODEL")

    # Verify critical keys
    msd = model.state_dict()
    must_keys = [
        "model.point_proj.weight",
        "model.point_proj.bias",
        "model.pre_proj_adapter.0.weight",
        "model.pre_proj_adapter.0.bias",
    ]
    for k in must_keys:
        if k in full_sd and k in msd:
            same = tuple(full_sd[k].shape) == tuple(msd[k].shape)
            logger.info(f"[LOAD_VERIFY] {k} present_in_ckpt=True present_in_model=True shape_match={same} shape={tuple(msd[k].shape)}")
        else:
            logger.warning(f"[LOAD_VERIFY] {k} present_in_ckpt={k in full_sd} present_in_model={k in msd}")

    model.eval().to(device)

    # keep point path fp32 stable (same habit as your eval before)
    model.get_model().point_backbone.float()
    try:
        model.get_model().pre_proj_adapter.float()
        model.get_model().point_proj.float()
    except Exception:
        pass

    # alpha sanity
    try:
        pb = model.get_model().point_backbone
        if hasattr(pb, "dec_adapter") and pb.dec_adapter is not None and getattr(pb.dec_adapter, "alpha", None) is not None:
            alpha = float(pb.dec_adapter.alpha.detach().cpu().item())
            gate = float(torch.sigmoid(pb.dec_adapter.alpha.detach()).cpu().item())
            logger.info(f"[SANITY] dec_adapter.alpha={alpha:.6f} sigmoid(alpha)={gate:.6f}")
    except Exception:
        pass

    if args.model_report:
        gm = model.get_model()
        logger.info("[MODEL_REPORT] has pre_proj_adapter: %s", hasattr(gm, "pre_proj_adapter"))
        logger.info("[MODEL_REPORT] has point_proj: %s", hasattr(gm, "point_proj"))
        logger.info("[MODEL_REPORT] has point_backbone: %s", hasattr(gm, "point_backbone"))
        logger.info("[MODEL_REPORT] use_enc_adapter=%s use_dec_adapter=%s", bool(int(args.use_enc_adapter)), bool(int(args.use_dec_adapter)))

    return model, tokenizer, amp_dtype


# ------------------------- scoring -------------------------
@torch.no_grad()
def score_labels_by_logprob_batch(
    model,
    tokenizer,
    prompt_text: str,
    labels: List[str],
    point_clouds: torch.Tensor,
    chunk_k: int = 8,
    use_amp: bool = True,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    device = point_clouds.device
    B = point_clouds.shape[0]
    K = len(labels)

    prompt_prefix = prompt_text + " "
    prefix_len = int(tokenizer([prompt_prefix], return_tensors="pt").input_ids.shape[1])

    scores = torch.zeros((B, K), device=device, dtype=torch.float32)

    full_texts = [prompt_prefix + lb for lb in labels]
    tok_all = tokenizer(full_texts, return_tensors="pt", padding=True)
    ids_all = tok_all.input_ids.to(device)
    attn_all = tok_all.attention_mask.to(device)

    valid_pos_per_k: List[torch.Tensor] = []
    for k in range(K):
        valid = (attn_all[k] == 1).nonzero(as_tuple=False).squeeze(1)
        valid = valid[valid >= max(1, prefix_len)]
        valid_pos_per_k.append(valid)

    use_cuda_amp = bool(use_amp and device.type == "cuda" and amp_dtype is not None and amp_dtype != torch.float32)
    amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_cuda_amp else nullcontext()

    for b in range(B):
        pc = point_clouds[b:b+1]  # [1, N, C]
        for s in range(0, K, chunk_k):
            e = min(K, s + chunk_k)
            ids_k = ids_all[s:e]
            attn_k = attn_all[s:e]
            kk = e - s

            with amp_ctx:
                out = model(
                    input_ids=ids_k,
                    attention_mask=attn_k,
                    point_clouds=pc.expand(kk, -1, -1),
                    use_cache=False,
                )
                logits = out.logits  # [kk, L, V]

            logp = F.log_softmax(logits.float(), dim=-1)

            for local_i, k in enumerate(range(s, e)):
                valid = valid_pos_per_k[k]
                if valid.numel() == 0:
                    scores[b, k] = -1e9
                    continue
                ids = ids_all[k]
                lp_sum = logp[local_i, valid - 1, ids[valid]].sum()
                scores[b, k] = (lp_sum / float(valid.numel())).float()

            del out, logits, logp
    return scores


@torch.no_grad()
def point_influence_sanity(
    model,
    tokenizer,
    prompt_text: str,
    labels: List[str],
    point_cloud: torch.Tensor,
    sanity_k: int = 8,
    chunk_k: int = 4,
    amp_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Dict[str, float]:
    device = point_cloud.device
    pc = point_cloud[:1] if point_cloud.ndim == 3 else point_cloud

    labels_sub = labels[:sanity_k]
    K = len(labels_sub)

    prompt_prefix = prompt_text + " "
    prefix_len = int(tokenizer([prompt_prefix], return_tensors="pt").input_ids.shape[1])

    full_texts = [prompt_prefix + lb for lb in labels_sub]
    tok = tokenizer(full_texts, return_tensors="pt", padding=True)
    ids_all = tok.input_ids.to(device)
    attn_all = tok.attention_mask.to(device)

    valid_pos_per_k: List[torch.Tensor] = []
    for k in range(K):
        valid = (attn_all[k] == 1).nonzero(as_tuple=False).squeeze(1)
        valid = valid[valid >= max(1, prefix_len)]
        valid_pos_per_k.append(valid)

    gm = model.get_model()
    pb = getattr(gm, "point_backbone", None)
    if pb is None:
        return {"mean_abs_dlogits": 0.0, "top1_changed_rate": 0.0, "mean_abs_dscores": 0.0}

    orig_forward = pb.forward

    def _forward_zero(point_clouds: torch.Tensor):
        out = orig_forward(point_clouds)
        return (torch.zeros_like(out[0]), out[1], out[2], out[3])

    use_cuda_amp = bool(device.type == "cuda" and amp_dtype is not None and amp_dtype != torch.float32)
    amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_cuda_amp else nullcontext()

    def _run_scores_and_logits(zeroed: bool):
        scores = torch.zeros((K,), device=device, dtype=torch.float32)
        dlogits_sum = 0.0
        dlogits_cnt = 0

        for s in range(0, K, chunk_k):
            e = min(K, s + chunk_k)
            ids_k = ids_all[s:e]
            attn_k = attn_all[s:e]
            kk = e - s

            # normal logits
            pb.forward = orig_forward
            with amp_ctx:
                out1 = model(
                    input_ids=ids_k,
                    attention_mask=attn_k,
                    point_clouds=pc.expand(kk, -1, -1),
                    use_cache=False,
                )
            logits1 = out1.logits.detach().float()

            # zeroed logits
            pb.forward = _forward_zero
            with amp_ctx:
                out2 = model(
                    input_ids=ids_k,
                    attention_mask=attn_k,
                    point_clouds=pc.expand(kk, -1, -1),
                    use_cache=False,
                )
            logits2 = out2.logits.detach().float()

            dlogits_sum += (logits1 - logits2).abs().mean().item()
            dlogits_cnt += 1

            logp = F.log_softmax((logits2 if zeroed else logits1), dim=-1)

            for local_i, k in enumerate(range(s, e)):
                valid = valid_pos_per_k[k]
                if valid.numel() == 0:
                    scores[k] = -1e9
                    continue
                ids = ids_all[k]
                lp_sum = logp[local_i, valid - 1, ids[valid]].sum()
                scores[k] = (lp_sum / float(valid.numel())).float()

            del out1, out2, logits1, logits2, logp

        return scores, (dlogits_sum / max(1, dlogits_cnt))

    s_normal, dlogits = _run_scores_and_logits(zeroed=False)
    s_zero, _ = _run_scores_and_logits(zeroed=True)

    pb.forward = orig_forward

    mean_abs_dscores = (s_normal - s_zero).abs().mean().item()
    top1_normal = int(torch.argmax(s_normal).item())
    top1_zero = int(torch.argmax(s_zero).item())
    top1_changed = 1.0 if top1_normal != top1_zero else 0.0

    return {
        "mean_abs_dlogits": float(dlogits),
        "mean_abs_dscores": float(mean_abs_dscores),
        "top1_changed_rate": float(top1_changed),
        "top1_normal": float(top1_normal),
        "top1_zero": float(top1_zero),
    }


def main(args):
    model, tokenizer, amp_dtype = init_model_internal(args)

    dataset = ModelNet40DirCompat(
        root=args.modelnet_root,
        split=args.split,
        npoints=args.npoints,
        subset_nums=args.subset_nums,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    question = PROMPT_LISTS[int(args.prompt_index)]
    prompt, _ = _build_prompt(args.conv_mode, question)
    prompt = _replace_point_token(prompt, args.point_token_len)

    if args.debug_point:
        logger.info(f"[DEBUG] token counts: {_count_tokens(tokenizer, prompt)}")
        logger.info(f"[DEBUG] prompt snippet:\n{prompt[:400]}...")

    device = next(model.parameters()).device
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- SANITY: point influence ----------
    if args.sanity_point_influence:
        try:
            first = next(iter(dataloader))
            pc0 = first["point_clouds"].to(device)
            pc0_1 = pc0[:1] if pc0.ndim == 3 else pc0
            sanity = point_influence_sanity(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                labels=MODELNET40_CATEGORIES,
                point_cloud=pc0_1,
                amp_dtype=amp_dtype,
            )
            t1n = MODELNET40_CATEGORIES[int(sanity["top1_normal"])]
            t1z = MODELNET40_CATEGORIES[int(sanity["top1_zero"])]
            logger.info(
                "[SANITY_POINT_INFLUENCE] mean|Δlogits|=%.6g mean|Δscores|=%.6g top1_changed=%s (normal=%s zero=%s)",
                sanity["mean_abs_dlogits"],
                sanity["mean_abs_dscores"],
                bool(sanity["top1_changed_rate"] > 0.5),
                t1n,
                t1z,
            )
            if sanity["mean_abs_dlogits"] < 1e-5 and sanity["mean_abs_dscores"] < 1e-6:
                logger.warning("[SANITY_POINT_INFLUENCE] Δ almost zero -> point injection likely NOT affecting logits.")
        except Exception as e:
            logger.warning(f"[SANITY_POINT_INFLUENCE] failed: {e}")

    # ---------- EVAL ----------
    results = []
    correct = 0
    total = 0
    printed = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        pc = batch["point_clouds"].to(device)
        gt_name = _get_label_name(batch)

        scores = score_labels_by_logprob_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            labels=MODELNET40_CATEGORIES,
            point_clouds=pc,
            chunk_k=args.chunk_k,
            use_amp=not args.fp32,
            amp_dtype=amp_dtype,
        )  # [B, 40]

        pred_idx = scores.argmax(dim=1).tolist()
        pred_labels = [MODELNET40_CATEGORIES[j] for j in pred_idx]

        for bi in range(pc.shape[0]):
            pred = pred_labels[bi]
            gt = gt_name
            ok = (gt is not None) and (pred == gt)
            total += 1
            correct += int(ok)

            if printed < args.print_first_k:
                logger.info(f"[SAMPLE {printed:03d}] GT={gt if gt is not None else 'N/A'} | Pred={pred} | ok={ok}")
                printed += 1

            top5 = torch.topk(scores[bi], k=5).indices.tolist()
            results.append(
                {
                    "idx": int(i * args.batch_size + bi),
                    "gt": gt,
                    "pred": pred,
                    "ok": bool(ok),
                    "top5": [MODELNET40_CATEGORIES[j] for j in top5],
                    "scores_top5": [float(scores[bi, j].item()) for j in top5],
                }
            )

        if args.only_first_batch:
            break

    acc = (correct / total) if total > 0 else 0.0
    logger.info(f"✅ Accuracy: {correct}/{total} = {acc:.4f}  (zero_point_ab={bool(args.zero_point_ab)})")

    out_path = os.path.join(args.output_dir, f"ModelNet_classification_prompt{args.prompt_index}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "correct": correct,
                "total": total,
                "zero_point_ab": bool(args.zero_point_ab),
                "use_enc_adapter": int(args.use_enc_adapter),
                "use_dec_adapter": int(args.use_dec_adapter),
                "items": results,
            },
            f,
            indent=2,
        )
    logger.info(f"✅ Saved to: {out_path}")

    if args.sanity_point_influence and acc < 0.05:
        logger.warning(
            "[DIAG] Very low acc. If SANITY_POINT_INFLUENCE shows near-zero Δ, "
            "most likely point_proj/pre_proj_adapter were NOT trained or NOT loaded from adapter_model.bin."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, default="RunsenXu_graspnet_enc_dec_r03_bak/PointLLM_7B_v1.2")
    parser.add_argument("--my_checkpoint_dir", type=str, required=True)
    parser.add_argument("--grasp_config", type=str, required=True)
    parser.add_argument("--grasp_ckpt", type=str, required=True)
    parser.add_argument("--modelnet_root", type=str, required=True)

    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--point_token_len", type=int, default=513)
    parser.add_argument("--backbone_output_dim", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--subset_nums", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--conv_mode", type=str, default="vicuna_v1_1")

    # dtype flags
    parser.add_argument("--bf16", action="store_true", help="Use bf16 (default if no dtype flag).")
    parser.add_argument("--fp16", action="store_true", help="Use fp16.")
    parser.add_argument("--fp32", action="store_true", help="Force fp32.")

    parser.add_argument("--output_dir", type=str, default="evaluation_internal")
    parser.add_argument("--debug_point", action="store_true")
    parser.add_argument("--print_first_k", type=int, default=10)

    parser.add_argument("--print_grasp_stats", action="store_true")
    parser.add_argument("--only_first_batch", action="store_true")
    parser.add_argument("--zero_point_ab", action="store_true")

    # chunk size for labels (VRAM control)
    parser.add_argument("--chunk_k", type=int, default=8)

    # train-aligned toggles
    parser.add_argument("--use_enc_adapter", type=int, default=0, help="0/1")
    parser.add_argument("--use_dec_adapter", type=int, default=1, help="0/1")
    parser.add_argument("--enc_adapter_ratio", type=int, default=8)
    parser.add_argument("--dec_adapter_ratio", type=int, default=8)

    # enhanced sanities
    parser.add_argument("--ckpt_report", action="store_true", default=True)
    parser.add_argument("--model_report", action="store_true", default=True)
    parser.add_argument("--sanity_point_influence", action="store_true", default=True)

    # alias for old run scripts
    parser.add_argument("--point_effect_sanity", action="store_true")

    args = parser.parse_args()

    # default dtype
    if not args.fp16 and not args.fp32 and not args.bf16:
        args.bf16 = True

    if getattr(args, "point_effect_sanity", False):
        args.sanity_point_influence = True

    main(args)
