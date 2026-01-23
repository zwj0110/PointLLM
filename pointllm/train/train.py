from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from types import SimpleNamespace, MethodType
import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
import json

# 确保环境路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pointllm.train.pointllm_trainer import PointLLMTrainer
from pointllm import conversation as conversation_lib
from pointllm.model import *
from pointllm.data import make_object_point_data_module
from pointllm.utils import build_logger

# --- [全局定义] ---
DEFAULT_POINT_PATCH_TOKEN = "<point_patch>"
DEFAULT_POINT_START_TOKEN = "<point_start>"
DEFAULT_POINT_END_TOKEN = "<point_end>"
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")
    point_backbone: str = field(default="PointBERT")


@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet")
    anno_path: str = field(default=None)
    dataset_type: str = field(default="modelnet40")
    use_color: bool = field(default=True)
    pointnum: int = field(default=8192)
    data_debug_num: int = field(default=0)
    split_train_val: bool = field(default=False)
    split_ratio: float = field(default=0.9)
    point_token_len: int = field(default=513)
    mm_use_point_start_end: bool = field(default=False)
    point_backbone_config: Optional[Dict[str, Any]] = field(default=None)
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"])
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=1024)
    cache_dir: Optional[str] = field(default=None)
    fix_llm: bool = field(default=True)
    fix_pointnet: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=True)
    point_backbone_ckpt: str = field(default=None)

    use_grasp: bool = field(default=False)
    grasp_ckpt: Optional[str] = field(default=None)
    grasp_config: Optional[str] = field(default=None)
    grasp_feat_dim: int = field(default=0)
    grasp_grid_size: int = field(default=256)
    grasp_coord_range: str = field(default="sphere")
    grasp_use_fps: bool = field(default=True)
    grasp_num_group: int = field(default=0)

    distill_alpha: float = field(default=1.0)
    lambda_rec: float = field(default=1.0)
    lambda_rate: float = field(default=0.01)


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "params"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _clean_state_dict_prefix(sd: dict) -> dict:
    new_sd = {}
    for k, v in sd.items():
        name = k.replace("module.", "").replace("model.", "").replace("point_encoder.", "")
        new_sd[name] = v
    return new_sd


def _inject_dual_path_logic(model, training_args, data_args, logger):
    """
    注入 teacher + student (Grasp) 双路径：
    - teacher: PointTransformer (f_teacher)
    - student: GeoResCompression (输出 [B,N,8] + R/D)
    重要：teacher 不挂到 student 上，避免被 state_dict 保存，保持轻量保存。
    """
    from pointllm.model.pointbert.point_encoder import PointTransformer
    from pointllm.pccai.models.architectures.grasp import GeoResCompression

    # 1) teacher 实例化
    backbone_yaml = "/home/zbellay/PycharmProjects/PointLLM/configs/PointTransformer_8192point_2layer.yaml"
    backbone_config = _load_yaml(backbone_yaml)["model"]

    teacher = PointTransformer(SimpleNamespace(**backbone_config)).to(training_args.device).float()
    teacher.load_state_dict(
        _clean_state_dict_prefix(
            _extract_state_dict(torch.load(training_args.point_backbone_ckpt, map_location="cpu"))
        ),
        strict=True,
    )
    teacher.eval()

    # 2) student 实例化
    raw_grasp_cfg = _load_yaml(training_args.grasp_config)
    student = GeoResCompression(
        raw_grasp_cfg.get("net_config", raw_grasp_cfg),
        SimpleNamespace(phase="train")
    ).to(training_args.device).float()

    student.load_state_dict(
        _extract_state_dict(torch.load(training_args.grasp_ckpt, map_location="cpu")),
        strict=False
    )

    # --- [日志 A] ---
    logger.info("🔍 [INIT DEBUG] 正在核对学生网络物理维度...")
    if hasattr(student, "eb_channel"):
        logger.info(f"   - 网络声明的 eb_channel: {student.eb_channel}")
    for name, param in student.named_parameters():
        if any(k in name for k in ["eb_layer", "output", "compress"]):
            logger.info(f"   - 关键权重 '{name}' 形状: {list(param.shape)}")

    # 3) 挂载到 LLM 模型中
    m = model.get_model()
    m.point_backbone = student

    # 4) 蒸馏对齐适配器 (8 -> 768)
    student.adapter1 = nn.Sequential(
        nn.Linear(student.eb_channel, 256),
        nn.ReLU(),
        nn.Linear(256, 768)
    ).to(training_args.device).float()

    # ✅ teacher 不挂到 student 属性上，避免 state_dict 包含 teacher
    def get_original_features(self, point_clouds):
        with torch.no_grad():
            return teacher(point_clouds.float())

    student.get_original_features = MethodType(get_original_features, student)

    # 5) student forward 包装（输入归一化 + shape 规整 + R/D 输出）
    def student_forward_wrapped(self, point_clouds, return_loss=True):
        do_log = not hasattr(self, "_logged_once")

        # 只取 xyz 并归一化到 unit sphere
        xyz = point_clouds[:, :, :3].float()
        centroid = torch.mean(xyz, dim=1, keepdim=True)
        xyz = xyz - centroid
        dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
        xyz = xyz / (dist.unsqueeze(-1) + 1e-6)

        if do_log:
            print(f"\n🚀 [RUNTIME DEBUG] Step 1: 输入 XYZ 归一化完成. 形状: {xyz.shape}")

        res = GeoResCompression.forward(self, xyz)

        if isinstance(res, dict):
            y_hat = res.get("y_hat")
            # likelihoods 可能是 dict/tensor，按你的原逻辑取 feats
            r_loss = res.get("likelihoods", {}).get("feats", torch.tensor(0.0, device=xyz.device))
            if torch.is_tensor(r_loss):
                r_loss = r_loss.mean()
            else:
                r_loss = torch.tensor(0.0, device=xyz.device)

            d_loss = res.get("d_loss", torch.tensor(0.0, device=xyz.device))
            if torch.is_tensor(d_loss):
                d_loss = d_loss.mean()
            else:
                d_loss = torch.tensor(0.0, device=xyz.device)
        else:
            y_hat, r_loss, d_loss = res if isinstance(res, tuple) else (res, 0.0, 0.0)

        if do_log and torch.is_tensor(y_hat):
            print(f"📊 [RUNTIME DEBUG] Step 2: GraspNet 原始输出 y_hat 形状: {tuple(y_hat.shape)}")
            print(f"📊 [RUNTIME DEBUG] Step 2: 元素总数: {y_hat.numel()}")
            nz_ratio = torch.count_nonzero(y_hat).item() / max(1, y_hat.numel())
            print(f"📊 [RUNTIME DEBUG] Step 2: 非零信号占比: {nz_ratio:.2%}")
            self._logged_once = True

        # 形状规整到 [B, N, 8]
        target_n, target_dim = int(data_args.point_token_len), 8
        batch_size = point_clouds.shape[0]

        if torch.is_tensor(y_hat):
            # 若 y_hat 塌陷成 [B*N] 或类似，强制恢复 channel
            if y_hat.numel() == batch_size * target_n:
                if do_log:
                    print("⚠️ [SIGNAL WARNING] 检测到特征维丢失（flatten），执行通道恢复到 8 维...")
                y_hat = y_hat.view(batch_size, target_n, 1).repeat(1, 1, target_dim)

            if y_hat.dim() == 2:
                y_hat = y_hat.view(batch_size, -1, y_hat.shape[-1])

            if y_hat.shape[1] != target_n:
                y_hat = F.interpolate(y_hat.transpose(1, 2).float(), size=target_n).transpose(1, 2)
        else:
            # 极端兜底
            y_hat = torch.zeros((batch_size, target_n, target_dim), device=point_clouds.device, dtype=point_clouds.dtype)

        # r_loss / d_loss 保证为 tensor
        if not torch.is_tensor(r_loss):
            r_loss = torch.tensor(float(r_loss), device=point_clouds.device)
        if not torch.is_tensor(d_loss):
            d_loss = torch.tensor(float(d_loss), device=point_clouds.device)

        return y_hat.to(point_clouds.dtype), r_loss, d_loss

    student.forward = MethodType(student_forward_wrapped, student)
    logger.info("✅ [OK] teacher+student 双路径注入完成（teacher 不参与保存）。")


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 0) 配置同步：point token 长度、输出维、patch token id ---
    full_p_cfg = {
        "point_token_len": int(data_args.point_token_len),
        "backbone_output_dim": 8,
        "default_point_patch_token": DEFAULT_POINT_PATCH_TOKEN,
        "mm_use_point_start_end": bool(data_args.mm_use_point_start_end),
        # 注意：point_patch_token_id 要在 tokenizer 初始化后补
    }
    data_args.point_backbone_config = full_p_cfg

    logger = build_logger(__name__, os.path.join(training_args.output_dir, "train.log"))

    # --- 1) Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False)
    tokenizer.add_tokens([DEFAULT_POINT_PATCH_TOKEN, DEFAULT_POINT_START_TOKEN, DEFAULT_POINT_END_TOKEN],
                         special_tokens=True)
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    # ✅ 把 <point_patch> 的 token id 写进 config（给 pointllm.py 精确替换使用）
    full_p_cfg["point_patch_token_id"] = tokenizer.convert_tokens_to_ids(DEFAULT_POINT_PATCH_TOKEN)

    # --- 2) 加载模型 ---
    dtype = torch.float16 if training_args.fp16 else torch.float32
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    # 同步配置（包含 patch_token_id）
    model.config.point_backbone_config = full_p_cfg
    model.get_model().point_backbone_config = full_p_cfg

    # --- 3) 架构注入 ---
    if training_args.use_grasp:
        _inject_dual_path_logic(model, training_args, data_args, logger)

    logger.info(f"🔍 [CONFIG] 最终验证配置: {json.dumps(full_p_cfg, indent=2)}")

    # --- 4) 参数冻结 / 解冻 ---
    model.requires_grad_(False)

    if training_args.tune_mm_mlp_adapter:
        trainable_keywords = ["adapter1", "pre_proj_adapter", "point_proj", "point_backbone"]
        for name, param in model.named_parameters():
            if any(key in name for key in trainable_keywords):
                param.requires_grad = True
                # 训练这些层用 fp32 更稳
                param.data = param.data.to(torch.float32)

        trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"🚀 总可训练参数量: {trainable_m:.2f} M")

    # --- 5) 数据 ---
    data_module = make_object_point_data_module(tokenizer=tokenizer, data_args=data_args)

    # --- 6) Trainer ---
    trainer = PointLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    logger.info("🏁 训练正式启动...")
    trainer.train(resume_from_checkpoint=None)
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
