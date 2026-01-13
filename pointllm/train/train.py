#  Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Optional, List
from types import SimpleNamespace

import os
import torch
import torch.nn as nn
import transformers

from pointllm.train.pointllm_trainer import PointLLMTrainer
from pointllm import conversation as conversation_lib
from pointllm.model import *
from pointllm.data import make_object_point_data_module
from pointllm.utils import *
from pointllm.data.utils import *

# * logger
from pointllm.utils import build_logger

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

def build_and_load_grasp_model(training_args, logger=None):
    """
    Build GRASP-Net GeoResCompression and load checkpoint.

    Required args:
      - training_args.grasp_config: path to GRASP net_config yaml (e.g. configs/net_config/grasp_dus1.yaml)
      - training_args.grasp_ckpt:   path to ckpt (e.g. grasp_ckpt/r03.pth)
      - training_args.grasp_codec_config: path to codec_config yaml (optional; only needed if your loader requires it)
    """
    import os
    import torch
    import yaml

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    grasp_config_path = getattr(training_args, "grasp_config", None)
    grasp_ckpt_path = getattr(training_args, "grasp_ckpt", None)

    if not grasp_config_path or not os.path.exists(grasp_config_path):
        raise FileNotFoundError(f"[GRASP] grasp_config not found: {grasp_config_path}")
    if not grasp_ckpt_path or not os.path.exists(grasp_ckpt_path):
        raise FileNotFoundError(f"[GRASP] grasp_ckpt not found: {grasp_ckpt_path}")

    # ---- load yaml configs ----
    with open(grasp_config_path, "r") as f:
        net_config = yaml.safe_load(f)

    codec_config_path = getattr(training_args, "grasp_codec_config", None)
    codec_config = None
    if codec_config_path:
        if not os.path.exists(codec_config_path):
            raise FileNotFoundError(f"[GRASP] grasp_codec_config not found: {codec_config_path}")
        with open(codec_config_path, "r") as f:
            codec_config = yaml.safe_load(f)

    # ---- build a lightweight "syntax" object expected by GRASP ----
    # GRASP code uses syntax.phase.lower() == 'train' in __init__
    class _Syntax:
        def __init__(self, phase="train", **kwargs):
            self.phase = phase
            for k, v in kwargs.items():
                setattr(self, k, v)

    # If your GRASP repo needs more fields, you can add them here:
    # e.g. base_only / phase / etc.
    syntax = _Syntax(phase="train")

    # ---- import GeoResCompression ----
    # IMPORTANT: adjust this import to match YOUR actual file location.
    # If you copied GRASP code into pointllm/pccai/models/architectures/grasp.py then:
    from ..pccai.models.architectures.grasp import GeoResCompression

    grasp_model = GeoResCompression(net_config=net_config, syntax=syntax)

    # ---- load checkpoint ----
    ckpt = torch.load(grasp_ckpt_path, map_location="cpu", weights_only=False)
    # Try common formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Clean 'module.' prefix if saved with DDP
    clean_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_sd[k] = v

    missing, unexpected = grasp_model.load_state_dict(clean_sd, strict=False)
    logger.info(f"[GRASP] Loaded ckpt: {grasp_ckpt_path}")
    logger.info(f"[GRASP] missing={len(missing)}, unexpected={len(unexpected)}")

    grasp_model.eval()  # usually we freeze it; training_args.fix_pointnet controls grads
    return grasp_model

def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for --grasp_config. Please `pip install pyyaml`.") from e
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_state_dict(ckpt):
    # common checkpoint layouts
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "params"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _clean_state_dict_prefix(sd: dict) -> dict:
    # remove common prefixes
    out = {}
    for k, v in sd.items():
        nk = k
        nk = nk.replace("module.", "")
        nk = nk.replace("model.", "")
        out[nk] = v
    return out


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")
    point_backbone: str = field(default="PointBERT")


@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet", metadata={"help": "Path to the training data."})
    anno_path: str = field(default=None, metadata={"help": "Path to the utterance data. If None, will use referit3d by defautl."})
    use_color: bool = field(default=False, metadata={"help": "Whether to use color."})
    data_debug_num: int = field(default=0, metadata={"help": "Number of data to use in debug mode. If larger than 0, use debug mode, else use the whole data"})
    split_train_val: bool = field(default=False, metadata={"help": "Whether to split train and val."})
    split_ratio: float = field(default=0.9, metadata={"help": "Ratio of train and val."})
    pointnum: int = field(default=8192, metadata={"help": "Number of points."})
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"], metadata={"help": "Conversation types to use."})
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # * can refer to https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArgument
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    model_debug: bool = field(default=False, metadata={"help": "Whether to use small model."})  # * whether to load checkpoints at the mo
    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(default=True, metadata={"help": "Whether to fix the PointNet."})

    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)

    # * for two stage training
    tune_mm_mlp_adapter: bool = field(default=True)  # * set True when pre-training, and false when fine-tuning
    stage_2: bool = field(default=False)  # * set True when fine-tuning
    pretrained_mm_mlp_adapter: Optional[str] = field(default=None)  # * path to the pre-trained projector & output_embed & input_embed
    detatch_point_token: bool = field(default=False)  # * deprecated

    # * point backbone ckpt path (PointBERT)
    point_backbone_ckpt: str = field(default=None)

    # ===== GRASP backbone options =====
    use_grasp: bool = field(default=False, metadata={"help": "Use GRASP backbone instead of PointBERT."})
    grasp_ckpt: Optional[str] = field(default=None, metadata={"help": "Path to GRASP checkpoint (GeoResCompression)."})
    grasp_config: Optional[str] = field(default=None, metadata={"help": "Path to GRASP net_config yaml."})
    grasp_codec_config: Optional[str] = field(default=None, metadata={"help": "Path to GRASP codec_config yaml."})
    grasp_feat_dim: int = field(default=0, metadata={"help": "Cg: feature dim output by grasp_model.res_enc."})
    grasp_grid_size: int = field(default=256, metadata={"help": "Voxel grid resolution for ME coords."})
    grasp_coord_range: str = field(default="sphere", metadata={"help": "sphere: [-1,1]->[0,1], unit: [0,1]."})
    grasp_use_fps: bool = field(default=True, metadata={"help": "Use FPS to sample tokens from coarse coords."})
    grasp_num_group: int = field(default=0, metadata={"help": "Override num_group (G). 0 means read from PointBERT yaml."})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _switch_to_grasp_backbone(model: "PointLLMLlamaForCausalLM", training_args: TrainingArguments, logger):
    """
    Replace PointBERT backbone with GRASP token backbone at runtime.
    This avoids trying to stuff a nn.Module into HF config.
    """
    if training_args.grasp_config is None:
        raise ValueError("[GRASP] --grasp_config is required when --use_grasp is set.")
    if training_args.grasp_ckpt is None:
        raise ValueError("[GRASP] --grasp_ckpt is required when --use_grasp is set.")
    if training_args.grasp_feat_dim <= 0:
        raise ValueError("[GRASP] --grasp_feat_dim (Cg) must be > 0 when --use_grasp is set.")

    # ---- build grasp GeoResCompression ----
    # !!! 改成你项目里 GeoResCompression 的实际 import 路径 !!!
    # 你贴的代码是 GeoResCompression 类，通常在 graspnet/pccai 项目里
    from ..pccai.models.architectures.grasp import GeoResCompression  # <- 你需要按实际路径修改

    net_cfg_all = _load_yaml(training_args.grasp_config)
    # 有些 yaml 外层会包一层，比如 {"net_config": {...}}
    net_config = net_cfg_all.get("net_config", net_cfg_all)

    # syntax 只要有 .phase 给 GeoResCompression 用即可
    syntax = SimpleNamespace(phase="train")

    grasp_model = GeoResCompression(net_config, syntax)

    ckpt = torch.load(training_args.grasp_ckpt, map_location="cpu", weights_only=False)
    sd = _extract_state_dict(ckpt)
    sd = _clean_state_dict_prefix(sd)
    missing, unexpected = grasp_model.load_state_dict(sd, strict=False)
    logger.info(f"[GRASP] Loaded grasp ckpt. missing={len(missing)}, unexpected={len(unexpected)}")

    grasp_model = grasp_model.to(training_args.device)
    grasp_model.eval()

    # ---- decide G (num_group) ----
    if training_args.grasp_num_group and training_args.grasp_num_group > 0:
        G = int(training_args.grasp_num_group)
        logger.info(f"[GRASP] Using grasp_num_group from args: {G}")
    else:
        # reuse PointBERT yaml num_group to keep prompt patch length consistent
        point_bert_config_name = getattr(model.config, "point_backbone_config_name", "PointTransformer_8192point_2layer")
        point_bert_config_addr = os.path.join(
            os.path.dirname(__file__),
            "..", "model", "pointbert", f"{point_bert_config_name}.yaml"
        )
        point_bert_config_addr = os.path.abspath(point_bert_config_addr)
        pb_cfg = cfg_from_yaml_file(point_bert_config_addr)
        G = int(pb_cfg.model.num_group)
        logger.info(f"[GRASP] Using PointBERT yaml num_group={G} from {point_bert_config_addr}")

    # ---- build token backbone wrapper ----
    from pointllm.model.grasp_backbone import GraspTokenBackbone, GraspBackboneArgs

    grasp_args = GraspBackboneArgs(
        num_group=G,
        grid_size=int(training_args.grasp_grid_size),
        coord_range=str(training_args.grasp_coord_range),
        use_fps=bool(training_args.grasp_use_fps),
        add_pos=True,
        cls_token=True,
    )

    Cg = int(training_args.grasp_feat_dim)
    token_backbone = GraspTokenBackbone(grasp_model=grasp_model, Cg=Cg, args=grasp_args)

    # ---- swap into PointLLM model ----
    m = model.get_model()
    m.point_backbone = token_backbone

    # update backbone config (used by tokenizer init + data_module)
    m.point_backbone_config = {
        "point_cloud_dim": 3,
        "backbone_output_dim": Cg,
        "project_output_dim": model.config.hidden_size,
        "point_token_len": G + 1,
        "mm_use_point_start_end": model.config.mm_use_point_start_end,
        "projection_hidden_layer": 0,
        "use_max_pool": False,
    }

    # rebuild projector to match Cg -> hidden
    m.point_proj = nn.Linear(Cg, model.config.hidden_size)

    # attach adapter (optional)
    try:
        from pointllm.model.transform_neck3d import TransformNeck3D
        m.point_backbone.transform_neck3d = TransformNeck3D(in_dim=Cg)
        logger.info("[GRASP] Attached TransformNeck3D adapter to GRASP backbone.")
    except Exception as e:
        logger.warning(f"[GRASP] Failed to attach TransformNeck3D adapter: {e}")
        m.point_backbone.transform_neck3d = None

    # mark flags for clarity
    model.config.point_backbone = "GRASP"
    model.config.grasp_grid_size = int(training_args.grasp_grid_size)
    model.config.grasp_coord_range = str(training_args.grasp_coord_range)
    model.config.grasp_feat_dim = Cg
    model.config.grasp_num_group = G

    logger.info(
        f"[GRASP] Switch complete. token_len={G+1}, Cg={Cg}, grid={training_args.grasp_grid_size}, range={training_args.grasp_coord_range}"
    )

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.log_level = "info"
    logger = build_logger(__name__, training_args.output_dir + "/train.log")

    # ================== DEVICE / GPU CHECK ==================
    cuda_ok = torch.cuda.is_available()
    num_gpu = torch.cuda.device_count() if cuda_ok else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok and num_gpu > 0 else "CPU"
    logger.info(f"[Device] torch.cuda.is_available={cuda_ok}, num_gpu={num_gpu}, gpu0={gpu_name}")
    logger.info(f"[Device] training_args.device={getattr(training_args, 'device', None)}")
    logger.info(f"[Env] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ================== 1) load ORIGINAL config (do NOT mutate to GRASP before from_pretrained) ==================
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # ================== 2) build model ==================
    if training_args.model_debug:
        # debug: build structure only
        model = PointLLMLlamaForCausalLM._from_config(config)
    else:
        # 1) build PointLLM structure (no weights)
        model = PointLLMLlamaForCausalLM._from_config(config)

        # 2) load a plain Llama/causalLM weights first (this won't hit your KeyError path)
        from transformers import AutoModelForCausalLM

        dtype = None
        if getattr(training_args, "bf16", False):
            dtype = torch.bfloat16
        elif getattr(training_args, "fp16", False):
            dtype = torch.float16

        base = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # 3) copy weights into your PointLLM wrapper (allow missing/unexpected keys)
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        logger.info(f"[LoadLLM] loaded base LLM weights. missing={len(missing)} unexpected={len(unexpected)}")

        del base


    logger.info(f"[Args] use_grasp={training_args.use_grasp}, grasp_feat_dim={training_args.grasp_feat_dim}")
    logger.info(f"[BeforeSwitch] backbone={model.get_model().point_backbone.__class__.__name__}")

    model.config.use_cache = False

    # ================== 3) switch to GRASP backbone AFTER loading ckpt ==================
    if training_args.use_grasp:
        _switch_to_grasp_backbone(model, training_args, logger)
    logger.info(f"[AfterSwitch] backbone={model.get_model().point_backbone.__class__.__name__}")
    logger.info(
        f"[AfterSwitch] backbone_output_dim={model.get_model().point_backbone_config.get('backbone_output_dim')}")

    # 强制确保真的切到 GRASP
    if training_args.use_grasp and model.get_model().point_backbone.__class__.__name__ != "GraspTokenBackbone":
        raise RuntimeError("[GRASP] switch failed: backbone is not GraspTokenBackbone")

    # ================== 4) freeze logic ==================
    if training_args.fix_llm:
        logger.info("LLM is fixed. Fix_llm=True")
        model.requires_grad_(False)
        model.get_model().fix_llm = True

        # keep projector trainable
        model.get_model().point_proj.requires_grad_(True)

        # for FSDP compatibility some code likes this True; fix_pointnet below will decide final
        model.get_model().point_backbone.requires_grad_(True)
    else:
        model.get_model().fix_llm = False
        logger.warning("LLM is trainable. Fix_llm=False")

    # point backbone freeze control
    if not training_args.fix_pointnet:
        logger.info("Point backbone is trainable. Fix_pointnet=False")
        model.get_model().fix_pointnet = False
    else:
        logger.info("Point backbone is fixed. Fix_pointnet=True")
        model.get_model().fix_pointnet = True
        if not training_args.stage_2:
            logger.info("Set requires_grad of point backbone to False")
            model.get_model().point_backbone.requires_grad_(False)

    # projector control
    if training_args.tune_mm_mlp_adapter:
        logger.info("Point projection layer is trainable (tune_mm_mlp_adapter=True).")
        model.get_model().point_proj.requires_grad_(True)
    else:
        logger.info("Point projection layer is fixed (tune_mm_mlp_adapter=False).")
        model.get_model().point_proj.requires_grad_(False)

    # ================== 5) tokenizer ==================
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0" or "v0" in model_args.model_name_or_path:
        raise ValueError("v0 is deprecated.")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    # ================== 6) init tokenizer/backbone config ==================
    if not training_args.stage_2:
        if training_args.use_grasp:
            logger.info("[GRASP] Skip load_point_backbone_checkpoint (PointBERT-only).")
        else:
            print(f"Default point_backbone_ckpt is {training_args.point_backbone_ckpt}.")
            model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)

        model.initialize_tokenizer_point_backbone_config(
            tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm
        )
    else:
        model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer=tokenizer)
    logger.info(f"[PostInit] backbone={model.get_model().point_backbone.__class__.__name__}")
    logger.info(f"[PostInit] point_token_len={model.get_model().point_backbone_config.get('point_token_len')}, "
                f"backbone_output_dim={model.get_model().point_backbone_config.get('backbone_output_dim')}")

    # ================== 7) log trainable params & final device ==================
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"[Trainable] tensors={len(trainable)}, params={sum(x[1] for x in trainable)}")
    for n, _ in trainable[:40]:
        logger.info(f"  [trainable] {n}")
    try:
        logger.info(f"[Device] model first param device: {next(model.parameters()).device}")
    except Exception as e:
        logger.warning(f"[Device] failed to read model param device: {e}")

    # ================== keep your original code below (data_args / data_module / trainer / train loop) ==================
    point_backbone_config = model.get_model().point_backbone_config

    data_args.point_token_len = point_backbone_config["point_token_len"]
    data_args.mm_use_point_start_end = point_backbone_config["mm_use_point_start_end"]
    data_args.point_backbone_config = point_backbone_config

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    "[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}".format(
                        len(params_no_grad), params_no_grad
                    )
                )
            else:
                print(
                    "[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)".format(
                        len(params_no_grad), ", ".join(params_no_grad[:10])
                    )
                )
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
            )

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop("use_orig_params", True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_object_point_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = PointLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
