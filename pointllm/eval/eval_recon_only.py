# -*- coding: utf-8 -*-
"""
Inference: PointLLM description generation on ModelNet40 (dir dataset)
Loads:
  1) Base PointLLM checkpoint (HF / local dir)
  2) Recon+Rate training RunsenXu_graspnet_enc_dec_r04: adapter_model.bin (tunable tensors)

Outputs:
  - JSON with per-object generated description.

Key notes:
- adapter_model.bin is loaded with strict=False and we report matched/unmatched keys.
- This script DOES NOT inject GRASP reconstructor by default, because your request is:
  "PointLLM returns description".
  If your recon training replaced point_backbone with GraspReconWrapper, those adapter weights
  will only affect inference if the same modules exist in the inference graph.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from pointllm.utils import disable_torch_init
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.model.utils import KeywordsStoppingCriteria

# Use the same model class as your previous eval script
from pointllm.model import PointLLMLlamaForCausalLM

from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat


PROMPT_LISTS = [
    "What is this? Please describe it in detail.",
    "This is an object. Please describe it in detail."
]


def setup_logger():
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("infer_desc")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_load_adapter_bin(adapter_bin_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(adapter_bin_path, map_location="cpu")
    # your train script saves: tunable = {name: tensor}
    if isinstance(obj, dict):
        # sometimes people save {"model": state_dict} etc. handle lightly:
        if all(isinstance(k, str) for k in obj.keys()):
            # assume it's already a state_dict-like
            return obj
    raise ValueError(f"adapter_model.bin format not recognized: type={type(obj)}")


def load_dataset(modelnet_root: str, split: str, npoints: int, subset_nums: int, use_color: bool):
    ds = ModelNet40DirCompat(
        root=modelnet_root,
        split=split,
        npoints=npoints,
        cache_npy=True,
        subset_nums=subset_nums,
        use_color=use_color,
    )
    return ds


def get_dataloader(dataset, batch_size: int, num_workers: int):
    # Keep shuffle=False to keep object_id stable
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def init_pointllm_with_adapter(
    base_model_path: str,
    adapter_bin_path: str,
    device: torch.device,
    dtype: torch.dtype,
    logger: logging.Logger,
    point_adapter_ckpt_for_projector: str = None,
):
    """
    Loads PointLLM base model WITHOUT AutoConfig (avoids KeyError: 'pointllm'),
    then writes custom fields onto model.config.
    Also loads recon_rate adapter_model.bin (tunable tensors) with strict=False.
    """
    disable_torch_init()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    tokenizer.add_tokens(["<point_patch>", "<point_start>", "<point_end>"], special_tokens=True)

    # ✅ DO NOT call AutoConfig here (it raises KeyError: 'pointllm')
    # Instead, load model directly (PointLLM class handles config/remote code internally)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=False,
        torch_dtype=dtype if device.type != "cpu" else torch.float32,
        trust_remote_code=True,
    )

    # resize embeddings after adding tokens
    try:
        model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        logger.warning(f"resize_token_embeddings failed (can be ok): {e}")

    model = model.to(device)
    model.eval()

    # ---- write custom fields onto model.config (replacing your old cfg.xxx approach) ----
    try:
        model.config.point_backbone = getattr(model.config, "point_backbone", "PointBERT")
        model.config.point_backbone_ckpt = getattr(model.config, "point_backbone_ckpt", None)
        model.config.mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
        model.config.fix_pointnet = getattr(model.config, "fix_pointnet", True)

        # optional old projector adapter ckpt
        if point_adapter_ckpt_for_projector is not None:
            model.config.point_adapter_ckpt = point_adapter_ckpt_for_projector
    except Exception as e:
        logger.warning(f"Setting custom config fields failed: {e}")

    # load projector adapter if supported (old PointLLM style)
    if hasattr(model, "load_point_adapter") and getattr(model.config, "point_adapter_ckpt", None):
        try:
            model.load_point_adapter()  # uses model.config.point_adapter_ckpt internally
            logger.info(f"✅ load_point_adapter() done (point_adapter_ckpt={model.config.point_adapter_ckpt})")
        except Exception as e:
            logger.warning(f"load_point_adapter() failed: {e}")

    # initialize point_backbone_config on model
    if hasattr(model, "initialize_tokenizer_point_backbone_config_wo_embedding"):
        try:
            model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
            logger.info("✅ initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer) done")
        except Exception as e:
            logger.warning(f"initialize_tokenizer_point_backbone_config_wo_embedding failed: {e}")

    # ---- load recon_rate adapter_model.bin (tunable tensors) ----
    adapter_loaded = False
    matched = 0
    total = 0

    if adapter_bin_path and os.path.isfile(adapter_bin_path):
        sd = safe_load_adapter_bin(adapter_bin_path)
        total = len(sd)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        adapter_loaded = True

        matched = total - len(unexpected)

        logger.info("=" * 80)
        logger.info(f"✅ Loaded recon_rate adapter_model.bin: {adapter_bin_path}")
        logger.info(f"  tensors in adapter bin: {total}")
        logger.info(f"  matched tensors loaded : {matched}")
        logger.info(f"  unexpected (not used)  : {len(unexpected)}")
        logger.info(f"  missing (model extra)  : {len(missing)}  (normal; model is huge)")
        if len(unexpected) > 0:
            logger.info("  Sample unexpected keys:")
            for k in unexpected[:30]:
                logger.info(f"    - {k}")
            if len(unexpected) > 30:
                logger.info(f"    ... ({len(unexpected)-30} more)")
        logger.info("=" * 80)
    else:
        logger.warning(f"adapter_bin_path not found, skip: {adapter_bin_path}")

    return model, tokenizer, adapter_loaded, matched, total



def build_prompt(conv, prompt_text: str, model) -> str:
    """
    Insert point tokens according to model.get_model().point_backbone_config.
    """
    point_backbone_config = getattr(model.get_model(), "point_backbone_config", None)
    if point_backbone_config is None:
        # fallback: no point token injection
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    point_token_len = int(point_backbone_config.get("point_token_len", 64))
    default_point_patch_token = point_backbone_config.get("default_point_patch_token", "<point_patch>")
    mm_use_point_start_end = bool(point_backbone_config.get("mm_use_point_start_end", False))

    if mm_use_point_start_end:
        default_point_start_token = point_backbone_config.get("default_point_start_token", "<point_start>")
        default_point_end_token = point_backbone_config.get("default_point_end_token", "<point_end>")
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + "\n" + prompt_text
    else:
        qs = default_point_patch_token * point_token_len + "\n" + prompt_text

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.inference_mode()
def generate_descriptions(
    model,
    tokenizer,
    conv_mode: str,
    dataloader,
    prompt_index: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    logger: logging.Logger,
):
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    prompt_text = PROMPT_LISTS[prompt_index]
    prompt = build_prompt(conv, prompt_text, model)

    inputs = tokenizer([prompt], return_tensors=None)
    input_ids_ = torch.as_tensor(inputs.input_ids).to(device)

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    results = {
        "prompt": prompt_text,
        "conv_mode": conv_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "results": []
    }

    for batch in tqdm(dataloader, desc="Generating"):
        point_clouds = batch["point_clouds"].to(device)

        # match model dtype for point path
        # (many point backbones expect fp32, but your previous eval used model.dtype)
        # We'll follow your previous style but keep safe fallback:
        try:
            point_clouds = point_clouds.to(dtype)
        except Exception:
            point_clouds = point_clouds.float()

        labels = batch.get("labels", None)
        label_names = batch.get("label_names", None)
        indice = batch.get("indice", None)

        B = point_clouds.shape[0]
        input_ids = input_ids_.repeat(B, 1)

        output_ids = model.generate(
            input_ids=input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
        )

        input_len = input_ids.shape[1]
        texts = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
        texts = [t.strip() for t in texts]

        for i in range(B):
            obj = {
                "object_id": int(indice[i].item()) if indice is not None else None,
                "model_output": texts[i],
            }
            if labels is not None:
                obj["ground_truth"] = int(labels[i].item())
            if label_names is not None:
                obj["label_name"] = label_names[i]
            results["results"].append(obj)

    return results


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Base PointLLM ckpt (HF name or local dir), e.g. RunsenXu_graspnet_enc_dec_r03_bak/PointLLM_7B_v1.2")
    parser.add_argument("--adapter_bin_path", type=str, required=True,
                        help="Recon+Rate RunsenXu_graspnet_enc_dec_r04 adapter_model.bin, e.g. checkpoints/pointllm_recon_rate_run_r03/adapter_model.bin")

    # optional: if you still want to load your projector adapter (old eval style)
    parser.add_argument("--point_projector_ckpt", type=str, default=None,
                        help="Optional: cfg.point_adapter_ckpt (e.g. output_r04_projector/adapter_best.pth)")

    parser.add_argument("--modelnet_root", type=str, required=True,
                        help="ModelNet40 dir root (dir compat)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--subset_nums", type=int, default=-1)
    parser.add_argument("--use_color", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--prompt_index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1_1")

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", default=True)

    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    device = get_device()
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        dtype = torch.float16

    # RunsenXu_graspnet_enc_dec_r04 dir
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.abspath(os.getcwd()), "evaluation_desc")
    os.makedirs(args.output_dir, exist_ok=True)

    out_json = os.path.join(
        args.output_dir,
        f"ModelNet_desc_prompt{args.prompt_index}_{args.split}.json"
    )

    logger.info(f"device={device}, dtype={dtype}")
    logger.info(f"base_model_path={args.base_model_path}")
    logger.info(f"adapter_bin_path={args.adapter_bin_path}")
    logger.info(f"modelnet_root={args.modelnet_root}, split={args.split}, npoints={args.npoints}")

    # dataset
    ds = load_dataset(
        modelnet_root=args.modelnet_root,
        split=args.split,
        npoints=args.npoints,
        subset_nums=args.subset_nums,
        use_color=args.use_color,
    )
    dl = get_dataloader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    model, tokenizer, adapter_loaded, matched, total = init_pointllm_with_adapter(
        base_model_path=args.base_model_path,
        adapter_bin_path=args.adapter_bin_path,
        device=device,
        dtype=dtype,
        logger=logger,
        point_adapter_ckpt_for_projector=args.point_projector_ckpt,
    )

    logger.info(f"adapter_loaded={adapter_loaded}, matched={matched}/{total}")

    # generate
    results = generate_descriptions(
        model=model,
        tokenizer=tokenizer,
        conv_mode=args.conv_mode,
        dataloader=dl,
        prompt_index=args.prompt_index,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        logger=logger,
    )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Saved: {out_json}")


if __name__ == "__main__":
    main()
