# -*- coding: utf-8 -*-
"""
Eval PointLLM description on ModelNet40, with optional GRASP (encoder+decoder) pre-processing.

IMPORTANT: This file aligns PointLLM init/prompt with the ORIGINAL author eval_modelnet_cls.py:
- DO NOT use AutoConfig/PretrainedConfig injections (may create mismatched modules).
- DO NOT load external projector ckpt by default.
- Use model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer).
- Build prompt using default_point_patch_token * point_token_len (same as author).

Modes:
- grasp_mode=none:        raw points -> PointLLM
- grasp_mode=encdec:      points -> GRASP encode+decode -> xyz_hat -> PointLLM
- grasp_mode=encdec_adapter: encdec + load adapter_model.bin into GRASP core (your recon adapters)

Extra sanities:
- print count of <point_patch> tokens in prompt
- point influence test: mean|Δlogits| between real pc and zero pc
"""

import os
import sys
import json
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM

try:
    from pointllm.data import ModelNet
except Exception:
    ModelNet = None

from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat

# timing & stats (keep)
from pointllm.eval.timing_utils import attach_timers, DeviceTimer
from pointllm.eval.model_stats import (
    params_by_module, group_sum,
    flops_by_module, pretty_print_flops,
)

# GRASP bridge
from pointllm.grasp_bridge import build_grasp_bridge


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


# ---------------- utils ----------------

def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
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


@torch.no_grad()
def sanity_point_influence(model, input_ids_1, pc_1):
    """
    Compare next-token logits with real pc vs zero pc.
    If points are used, logits should differ.
    """
    model.eval()
    out_real = model(input_ids=input_ids_1, point_clouds=pc_1, use_cache=False, return_dict=True)
    out_zero = model(input_ids=input_ids_1, point_clouds=torch.zeros_like(pc_1), use_cache=False, return_dict=True)
    delta = (out_real.logits[:, -1, :] - out_zero.logits[:, -1, :]).abs().mean().item()
    logger.info(f"[SANITY] point influence mean|Δlogits| = {delta:.6f}")
    if delta < 1e-5:
        logger.warning("[SANITY WARN] Δlogits ~ 0. The model is NOT using point_clouds (or point tokens not parsed).")


# ---------------- PointLLM init (ALIGN WITH AUTHOR) ----------------

def init_pointllm_for_desc(args):
    disable_torch_init()
    model_name = os.path.expanduser(args.base_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp16":
        llm_dtype = torch.float16
    elif args.dtype == "bf16":
        llm_dtype = torch.bfloat16
    else:
        llm_dtype = torch.float32

    logger.info(f"device={device}, dtype={llm_dtype}")
    logger.info(f"base_model_path={model_name}")

    # align with author's tokenizer init (no use_fast=False requirement)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # align with author's model init
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        use_cache=True,
        torch_dtype=llm_dtype if device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(device)

    # MUST: init point_backbone_config
    if hasattr(model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    else:
        logger.warning("⚠️ model missing initialize_tokenizer_point_backbone_config_wo_embedding; point tokens may not work.")

    conv = conv_templates["vicuna_v1_1"].copy()
    return model, tokenizer, conv


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


# ---------------- prompt / generation ----------------

def build_prompt_like_author(conv, model, prompt_index: int):
    """
    EXACT author style:
      qs = patch_token * point_token_len + '\n' + question
    No space separation.
    """
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    pbc = model.get_model().point_backbone_config
    point_token_len = int(pbc["point_token_len"])
    patch_tok = pbc["default_point_patch_token"]
    mm_use_point_start_end = bool(pbc.get("mm_use_point_start_end", False))

    if mm_use_point_start_end:
        start_tok = pbc["default_point_start_token"]
        end_tok = pbc["default_point_end_token"]
        qs = start_tok + (patch_tok * point_token_len) + end_tok + "\n" + qs
    else:
        qs = (patch_tok * point_token_len) + "\n" + qs

    conv2 = conv.copy()
    conv2.append_message(conv2.roles[0], qs)
    conv2.append_message(conv2.roles[1], None)
    prompt = conv2.get_prompt()
    return prompt, stop_str, patch_tok, point_token_len


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

    # base model
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
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    # RunsenXu_graspnet_enc_dec_r04
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    # sample printing
    parser.add_argument("--print_samples", action="store_true", default=True)
    parser.add_argument("--sample_every", type=int, default=50)
    parser.add_argument("--num_sample_print", type=int, default=2)

    # GRASP modes
    parser.add_argument("--grasp_mode", type=str, default="none",
                        choices=["none", "encdec", "encdec_adapter"])
    parser.add_argument("--grasp_ckpt", type=str, default=None)
    parser.add_argument("--grasp_config", type=str, default=None)
    parser.add_argument("--adapter_bin_path", type=str, default=None)
    parser.add_argument("--voxel_size", type=float, default=0.01)

    # optional encoder adapter in GRASP
    parser.add_argument("--use_enc_adapter", action="store_true", default=False)
    parser.add_argument("--enc_adapter_ratio", type=int, default=8)

    # recon metrics
    parser.add_argument("--compute_recon_metrics", action="store_true", default=False)
    parser.add_argument("--chamfer_sample_n", type=int, default=2048)

    # sanities
    parser.add_argument("--sanity_point_influence", action="store_true", default=True)
    parser.add_argument("--sanity_patch_count", action="store_true", default=True)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}")

    # RunsenXu_graspnet_enc_dec_r04 paths
    if args.output_dir is None:
        tag = f"grasp_{args.grasp_mode}"
        args.output_dir = os.path.join(args.base_model_path.replace("/", "_"), f"evaluation_{tag}")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_file is None:
        args.output_file = f"ModelNet_desc_prompt{args.prompt_index}_{args.grasp_mode}.json"

    out_path = os.path.join(args.output_dir, args.output_file)
    logger.info(f"output: {out_path}")

    # dataset
    ds = load_dataset(args)
    dl = get_dataloader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # init pointllm (author-aligned)
    model, tokenizer, conv = init_pointllm_for_desc(args)
    timers, _hooks = attach_timers(model, tokenizer)
    logger.info("[Timing] timers attached (tokenizer / point_encoder / projector).")

    # init GRASP bridge if needed
    grasp_bridge = None
    if args.grasp_mode != "none":
        assert args.grasp_ckpt and args.grasp_config, "--grasp_mode!=none requires --grasp_ckpt and --grasp_config"
        adapter_bin = args.adapter_bin_path if args.grasp_mode == "encdec_adapter" else None
        if args.grasp_mode == "encdec_adapter":
            assert adapter_bin is not None, "--grasp_mode=encdec_adapter requires --adapter_bin_path"

        grasp_bridge, matched, total = build_grasp_bridge(
            grasp_config=args.grasp_config,
            grasp_ckpt=args.grasp_ckpt,
            device=device,
            voxel_size=args.voxel_size,
            npoints=args.npoints,
            adapter_bin_path=adapter_bin,
            use_enc_adapter=args.use_enc_adapter,
            enc_adapter_ratio=args.enc_adapter_ratio,
        )
        logger.info(f"✅ GRASP bridge built. adapter_bin={'yes' if adapter_bin else 'no'} matched={matched} total_in_bin={total}")

        # warmup once
        try:
            grasp_bridge.eval()
            with torch.no_grad():
                b0 = next(iter(dl))
                _ = grasp_bridge(b0["point_clouds"][:1].to(device).float())
            logger.info("✅ GRASP bridge warmup ok.")
        except Exception as e:
            logger.warning(f"GRASP bridge warmup failed: {e}")

    # params summary
    core = model.get_model()
    rows, _ = params_by_module(core)
    prefixes = {
        "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
        "Projector": ["point_proj", "mm_projector", "projector"],
        "LLM": ["language_model", "model", "transformer", "lm"],
    }
    grouped_params = group_sum({name: n for name, _, n in rows}, prefixes)
    logger.info("Grouped Params (M): " + str({k: v / 1e6 for k, v in grouped_params.items()}))

    # build prompt once (author style)
    prompt, stop_str, patch_tok, point_token_len = build_prompt_like_author(conv, model, args.prompt_index)

    tok_timer = timers.get("tokenizer", DeviceTimer("tokenizer"))
    with tok_timer:
        tok = tokenizer([prompt])
    timers["tokenizer"] = tok_timer

    input_ids_ = torch.as_tensor(tok.input_ids).to(model.device)
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    # SANITY: patch count
    if args.sanity_patch_count:
        patch_id = tokenizer.convert_tokens_to_ids(patch_tok)
        ids_cpu = torch.as_tensor(tok.input_ids)
        cnt_patch = (ids_cpu == patch_id).sum().item()
        logger.info(f"[SANITY] patch_tok='{patch_tok}' id={patch_id} count_in_prompt={cnt_patch} expect(point_token_len)={point_token_len}")
        if cnt_patch < max(8, point_token_len // 8):
            logger.warning("[SANITY WARN] patch token count seems too small. Point tokens may not be parsed.")

    # FLOPs (optional)
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

    # SANITY: point influence
    if args.sanity_point_influence:
        try:
            it_s = iter(dl)
            b0 = next(it_s)
            pc0 = b0["point_clouds"][:1].to(device).float()
            pc0 = _normalize_unit_sphere(pc0).to(device).to(model.dtype)
            ids0 = input_ids_[:1]
            sanity_point_influence(model, ids0, pc0)
        except Exception as e:
            logger.warning(f"[SANITY] point influence failed: {e}")

    # iterate + generate
    results = {
        "prompt": PROMPT_LISTS[args.prompt_index],
        "grasp_mode": args.grasp_mode,
        "results": []
    }

    pbar = tqdm(iter(dl), total=len(dl), desc="Generating")
    for step, batch in enumerate(pbar):
        pc = batch["point_clouds"].to(device).float()
        labels = batch.get("labels", None)
        label_names = batch.get("label_names", None)
        indice = batch.get("indice", None)

        chamfer_val = None
        rate_val = None

        if grasp_bridge is None:
            pc_used = pc.to(device).to(model.dtype)
        else:
            with torch.no_grad():
                xyz_hat, rate, _out = grasp_bridge(pc)
            pc_used = _normalize_unit_sphere(xyz_hat).to(device).to(model.dtype)

            if args.compute_recon_metrics:
                with torch.no_grad():
                    gt = _normalize_unit_sphere(pc)
                    ch = chamfer_l2_approx(_normalize_unit_sphere(xyz_hat), gt, sample_n=args.chamfer_sample_n)
                    chamfer_val = float(ch.detach().cpu().item())
                    if rate is not None:
                        rate_val = float(rate.detach().cpu().item())

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

        if args.print_samples and (step % max(1, args.sample_every) == 0):
            k = min(args.num_sample_print, bs)
            logger.info("=" * 80)
            logger.info(f"[SAMPLE] step={step}, grasp_mode={args.grasp_mode}, chamfer={chamfer_val}, rate={rate_val}")
            for i in range(k):
                oid = int(indice[i].item()) if indice is not None else (step * bs + i)
                gtname = label_names[i] if label_names is not None else "N/A"
                logger.info(f"  object_id={oid}, gt={gtname}")
                logger.info(f"  output: {outs[i]}")
            logger.info("=" * 80)

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
