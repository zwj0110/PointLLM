import os
import sys
import glob
import argparse
import json
import yaml
import logging
from types import SimpleNamespace, MethodType

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# --- [1. 路径注入] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
pccai_root = os.path.join(project_root, "pointllm")

if project_root not in sys.path: sys.path.append(project_root)
if pccai_root not in sys.path: sys.path.append(pccai_root)

try:
    from pointllm.model import PointLLMLlamaForCausalLM
    from pointllm.pccai.models.architectures.grasp import GeoResCompression
    from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat
    from pointllm.model.utils import KeywordsStoppingCriteria
    from pointllm.conversation import conv_templates
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_POINT_PATCH_TOKEN = "<point_patch>"


def _load_yaml(path: str):
    with open(path, "r") as f: return yaml.safe_load(f)


def _pick_latest_bin(ckpt_dir: str) -> str:
    cand = os.path.join(ckpt_dir, "adapter_model.bin")
    if os.path.exists(cand): return cand
    all_bins = sorted(glob.glob(os.path.join(ckpt_dir, "*.bin")), key=os.path.getmtime, reverse=True)
    if not all_bins: raise FileNotFoundError(f"❌ 未找到权重文件: {ckpt_dir}")
    return all_bins[0]


def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
    xyz = point_clouds[:, :, :3].float()
    centroid = torch.mean(xyz, dim=1, keepdim=True)
    xyz = xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
    return xyz / (dist.unsqueeze(-1) + 1e-6)


def init_model_with_distill(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)
    tokenizer.add_tokens([DEFAULT_POINT_PATCH_TOKEN], special_tokens=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token

    # --- 2. 加载基础模型 ---
    logger.info(f"正在从 {args.base_model_path} 加载基础模型...")
    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # --- 3. 初始化 Grasp 学生网络 ---
    logger.info(f"正在初始化 Grasp 学生网络 (Config: {args.grasp_config})...")
    raw_cfg = _load_yaml(args.grasp_config)
    student = GeoResCompression(raw_cfg.get("net_config", raw_cfg), SimpleNamespace(phase="test")).to(device)
    student.eval()
    model.get_model().point_backbone = student

    # --- 4. 注入权重并打印详细 Key 映射 ---
    target_bin = _pick_latest_bin(args.my_checkpoint_dir)
    logger.info(f"正在读取权重文件: {target_bin}")
    state_dict = torch.load(target_bin, map_location="cpu")

    logger.info(f"权重文件中的前 5 个 Key 示例: {list(state_dict.keys())[:5]}")

    remapped_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        # 强制将 point_proj 映射到模型期望的路径
        if "point_proj" in new_k and not new_k.startswith("model."):
            new_k = "model." + new_k
        if "point_backbone" in new_k and not new_k.startswith("model."):
            new_k = "model." + new_k
        remapped_sd[new_k] = v

    msg = model.load_state_dict(remapped_sd, strict=False)

    # 打印关键层的状态
    proj_status = "✅ 已加载" if not any("point_proj" in k for k in msg.missing_keys) else "❌ 缺失"
    backbone_status = "✅ 已加载" if not any("point_backbone" in k for k in msg.missing_keys) else "❌ 缺失"
    logger.info(f"投影层加载状态: {proj_status}")
    logger.info(f"学生网络加载状态: {backbone_status}")
    if msg.unexpected_keys:
        logger.info(f"未匹配到的 Key (Unexpected): {msg.unexpected_keys[:5]}...")

    # --- 5. 增强版前向 Hook ---
    def student_forward_eval(self, point_clouds):
        xyz = _normalize_unit_sphere(point_clouds)
        res = GeoResCompression.forward(self, xyz)
        y_hat = res.get("y_hat") if isinstance(res, dict) else res[0]

        if y_hat.dim() == 2: y_hat = y_hat.view(xyz.shape[0], -1, y_hat.shape[-1])

        # 数值统计日志（仅在第一次调用或特征异常时打印）
        if not hasattr(self, '_logged_stats'):
            mean, std = y_hat.mean().item(), y_hat.std().item()
            logger.info(f"学生特征统计 - Mean: {mean:.4f}, Std: {std:.4f}, Shape: {list(y_hat.shape)}")
            self._logged_stats = True
            if std < 1e-5:
                logger.warning("⚠️ 学生特征方差极低，模型可能未收敛或致盲！")

        target_n = int(args.point_token_len)
        if y_hat.shape[1] != target_n:
            y_hat = y_hat.transpose(1, 2)
            y_hat = F.interpolate(y_hat, size=target_n, mode='linear', align_corners=False)
            y_hat = y_hat.transpose(1, 2)

        return (y_hat * float(args.signal_scale)).half()

    student.forward = MethodType(student_forward_eval, student)

    # 同步配置
    p_cfg = {
        "point_token_len": int(args.point_token_len),
        "backbone_output_dim": 8,
        "mm_use_point_start_end": False,
        "point_patch_token_id": tokenizer.convert_tokens_to_ids(DEFAULT_POINT_PATCH_TOKEN),
    }
    model.get_model().point_backbone_config = p_cfg
    model.config.point_backbone_config = p_cfg

    return model, tokenizer


def main(args):
    model, tokenizer = init_model_with_distill(args)
    model.eval()

    dataset = ModelNet40DirCompat(root=args.modelnet_root, split="test", npoints=8192)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    prompt = conv_templates["vicuna_v1_1"].copy()
    qs = "What is the category of this 3D object? Answer in one short sentence."
    prompt.append_message(prompt.roles[0], DEFAULT_POINT_PATCH_TOKEN * int(args.point_token_len) + "\n" + qs)
    prompt.append_message(prompt.roles[1], None)
    full_prompt = prompt.get_prompt()

    logger.info(f"推理 Prompt 样例 (前 50 字): {full_prompt[:50]}...")

    for i, batch in enumerate(tqdm(dataloader)):
        pc = batch["point_clouds"].to(model.device)
        input_ids = torch.as_tensor(tokenizer([full_prompt]).input_ids).to(model.device)
        if pc.shape[0] > 1: input_ids = input_ids.repeat(pc.shape[0], 1)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                point_clouds=pc.half(),
                do_sample=False,
                max_new_tokens=20,
            )

        for j in range(len(output_ids)):
            output_text = tokenizer.decode(output_ids[j, input_ids.shape[1]:], skip_special_tokens=True).strip()
            results.append({"label": batch["label_names"][j], "pred": output_text})
            if i < 3:
                logger.info(f"Sample {i} | GT: {batch['label_names'][j]} | Pred: {output_text}")

    out_p = os.path.join(args.my_checkpoint_dir, "eval_debug_results.json")
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"推理结束，结果存至: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="RunsenXu_graspnet_enc_dec_r03_bak/PointLLM_7B_v1.2")
    parser.add_argument("--my_checkpoint_dir", type=str, required=True)
    parser.add_argument("--grasp_config", type=str, required=True)
    parser.add_argument("--modelnet_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--point_token_len", type=int, default=513)
    parser.add_argument("--signal_scale", type=float, default=1.0)
    main(parser.parse_args())