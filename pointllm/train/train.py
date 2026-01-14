from dataclasses import dataclass, field
import pathlib
from typing import Optional, List
from types import SimpleNamespace
import sys
import os
import yaml
import torch
import torch.nn as nn
import transformers

# 确保项目根目录在路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pointllm.train.pointllm_trainer import PointLLMTrainer
from pointllm import conversation as conversation_lib
from pointllm.model import *
from pointllm.data import make_object_point_data_module
from pointllm.utils import *
from pointllm.data.utils import *
from pointllm.utils import build_logger

IGNORE_INDEX = -100


# --- 工具函数 ---
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
    out = {}
    for k, v in sd.items():
        nk = k.replace("module.", "").replace("model.", "")
        out[nk] = v
    return out


# --- 参数类 ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")
    point_backbone: str = field(default="PointBERT")


@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet")
    anno_path: str = field(default=None)
    use_color: bool = field(default=False)
    data_debug_num: int = field(default=0)
    split_train_val: bool = field(default=False)
    split_ratio: float = field(default=0.9)
    pointnum: int = field(default=8192)
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"])
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    model_debug: bool = field(default=False)
    fix_llm: bool = field(default=True)
    fix_pointnet: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=True)
    stage_2: bool = field(default=False)
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None)
    point_backbone_ckpt: str = field(default=None)

    # GRASP 选项
    use_grasp: bool = field(default=False)
    grasp_ckpt: Optional[str] = field(default=None)
    grasp_config: Optional[str] = field(default=None)
    grasp_feat_dim: int = field(default=0)
    grasp_grid_size: int = field(default=256)
    grasp_coord_range: str = field(default="sphere")
    grasp_use_fps: bool = field(default=True)
    grasp_num_group: int = field(default=0)


# --- 核心切换逻辑 ---
def _switch_to_grasp_backbone(model, training_args, logger):
    """替换 PointBERT 为 GRASP 并更新投影层"""
    if not training_args.grasp_config or not training_args.grasp_ckpt:
        raise ValueError("[GRASP] --grasp_config 和 --grasp_ckpt 必填")

    # 1. 加载 GRASP 架构 (GeoResCompression)
    from pointllm.pccai.models.architectures.grasp import GeoResCompression
    net_cfg_all = _load_yaml(training_args.grasp_config)
    net_config = net_cfg_all.get("net_config", net_cfg_all)
    grasp_model = GeoResCompression(net_config, SimpleNamespace(phase="train"))

    # 2. 加载权重
    ckpt = torch.load(training_args.grasp_ckpt, map_location="cpu", weights_only=False)
    sd = _clean_state_dict_prefix(_extract_state_dict(ckpt))
    missing, unexpected = grasp_model.load_state_dict(sd, strict=False)
    logger.info(f"[GRASP] 权重加载成功. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # 3. 确定 Token 数量 G
    if training_args.grasp_num_group > 0:
        G = training_args.grasp_num_group
    else:
        # 兜底方案：读取默认配置
        G = 512

        # 4. 创建 GraspTokenBackbone 包装器
    from pointllm.model.grasp_backbone import GraspTokenBackbone, GraspBackboneArgs
    grasp_args = GraspBackboneArgs(
        num_group=G,
        grid_size=training_args.grasp_grid_size,
        coord_range=training_args.grasp_coord_range,
        use_fps=training_args.grasp_use_fps,
        add_pos=True,
        cls_token=True,
    )

    Cg = training_args.grasp_feat_dim
    token_backbone = GraspTokenBackbone(grasp_model=grasp_model, Cg=Cg, args=grasp_args)

    # 5. 替换模型组件
    m = model.get_model()
    m.point_backbone = token_backbone.to(training_args.device)

    # 重新初始化 Projector (维度从 Cg 变为 LLM hidden_size)
    m.point_proj = nn.Linear(Cg, model.config.hidden_size).to(training_args.device)

    # 更新元数据
    m.point_backbone_config = {
        "point_cloud_dim": 3,
        "backbone_output_dim": Cg,
        "project_output_dim": model.config.hidden_size,
        "point_token_len": G + 1,
        "mm_use_point_start_end": getattr(model.config, "mm_use_point_start_end", False),
        "projection_hidden_layer": 0,
        "use_max_pool": False,
    }

    # 同步到 model.config 以便保存 Checkpoint 时生效
    model.config.point_backbone = "GRASP"
    model.config.backbone_output_dim = Cg
    model.config.point_token_len = G + 1

    logger.info(f"[GRASP] 架构切换完成. TokenLen: {G + 1}, Cg: {Cg}")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger = build_logger(__name__, os.path.join(training_args.output_dir, "train.log"))

    # 1. 初始化模型
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
    model = PointLLMLlamaForCausalLM._from_config(config)

    # 2. 加载基础 LLM 权重
    if not training_args.model_debug:
        dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
        base = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.load_state_dict(base.state_dict(), strict=False)
        del base

    # 3. 核心：切换 Backbone
    if training_args.use_grasp:
        _switch_to_grasp_backbone(model, training_args, logger)
    else:
        # 原有 PointBERT 加载逻辑
        if not training_args.stage_2:
            model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)

    # 4. 梯度冻结控制 (必须在切换架构后进行)
    if training_args.fix_llm:
        logger.info("冻结 LLM 权重")
        model.requires_grad_(False)
        model.get_model().fix_llm = True

    # Projector 梯度
    if training_args.tune_mm_mlp_adapter:
        logger.info("开启 Projector 训练")
        model.get_model().point_proj.requires_grad_(True)

    # Backbone 梯度
    if not training_args.fix_pointnet:
        logger.info("开启 Point Backbone 训练")
        model.get_model().point_backbone.requires_grad_(True)
        model.get_model().fix_pointnet = False
    else:
        model.get_model().point_backbone.requires_grad_(False)

    # 5. Tokenizer 处理
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    # 6. 数据与 Trainer
    point_config = model.get_model().point_backbone_config
    data_args.point_token_len = point_config["point_token_len"]
    data_args.mm_use_point_start_end = point_config["mm_use_point_start_end"]
    data_args.point_backbone_config = point_config

    data_module = make_object_point_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = PointLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # 7. 开始训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()