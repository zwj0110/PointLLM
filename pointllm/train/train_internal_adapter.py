# -*- coding: utf-8 -*-
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from types import SimpleNamespace
from transformers import AutoTokenizer

DEBUG = int(os.environ.get("POINTLLM_DEBUG", "0"))
LOCAL_FILES_ONLY = int(os.environ.get("LOCAL_FILES_ONLY", "1"))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pointllm.model.pointllm_internal import PointLLMLlamaForCausalLMInternal
from pointllm.data import make_object_point_data_module
from pointllm.utils import build_logger

IGNORE_INDEX = -100


def _extract_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return ckpt
    for k in ["model", "state_dict", "net", "params", "net_state_dict"]:
        if k in ckpt:
            return ckpt[k]
    return ckpt


def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
    xyz = point_clouds[:, :, :3].float()
    centroid = torch.mean(xyz, dim=1, keepdim=True)
    xyz = xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
    return xyz / (dist.unsqueeze(-1) + 1e-6)


def _get_modules_cfg(grasp_yaml: dict) -> dict:
    if isinstance(grasp_yaml, dict) and "modules" in grasp_yaml and isinstance(grasp_yaml["modules"], dict):
        return grasp_yaml["modules"]
    if isinstance(grasp_yaml, dict) and "net_config" in grasp_yaml and isinstance(grasp_yaml["net_config"], dict):
        return grasp_yaml["net_config"]
    return grasp_yaml


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")


@dataclass
class DataArguments:
    data_path: str = field(default="ScanNet")
    anno_path: str = field(default=None)
    dataset_type: str = field(default="modelnet40")
    use_color: bool = field(default=False)
    pointnum: int = field(default=8192)
    point_token_len: int = field(default=513)
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description"])
    is_multimodal: bool = True
    point_backbone_config: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=1024)
    point_backbone_ckpt: str = field(default=None)   # teacher (PointTransformer / PointBERT)
    grasp_ckpt: Optional[str] = field(default=None)
    grasp_config: Optional[str] = field(default=None)
    use_grasp: bool = field(default=True)

    # ---- multi-loss weights ----
    distill_alpha: float = field(default=1.0)  # feature distill
    lambda_rec: float = field(default=1.0)     # D_loss
    lambda_rate: float = field(default=0.01)   # R_loss
    lambda_ce: float = field(default=1.0)      # ✅ NEW: CE / LM loss weight

    # ---- adapters ----
    use_enc_adapter: bool = field(default=True)
    use_dec_adapter: bool = field(default=True)
    enc_adapter_ratio: int = field(default=8)
    dec_adapter_ratio: int = field(default=8)


def _inject_internal_adapter_logic(model, training_args, data_args, logger):
    from pointllm.model.pointbert.point_encoder import PointTransformer
    from pointllm.pccai.models.architectures.grasp import GeoResCompression
    from pointllm.pccai.models.modules.adapters import ResidualConv1dAdapter
    from pointllm.pccai.models.modules.adapters import MinkowskiAdapter

    if training_args.point_backbone_ckpt is None:
        raise ValueError("point_backbone_ckpt (teacher) is None")
    if training_args.grasp_ckpt is None or training_args.grasp_config is None:
        raise ValueError("grasp_ckpt / grasp_config must be provided")

    device = training_args.device

    # ---------------- Teacher (frozen) ----------------
    backbone_yaml = "/home/zbellay/PycharmProjects/PointLLM/configs/PointTransformer_8192point_2layer.yaml"
    backbone_config = yaml.safe_load(open(backbone_yaml))["model"]
    teacher = PointTransformer(SimpleNamespace(**backbone_config)).to(device).eval()
    teacher.load_state_dict(_extract_state_dict(training_args.point_backbone_ckpt), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False

    # ---------------- Student GRASP core ----------------
    grasp_yaml = yaml.safe_load(open(training_args.grasp_config))
    modules_cfg = _get_modules_cfg(grasp_yaml)
    grasp_core = GeoResCompression(modules_cfg, SimpleNamespace(phase="train")).to(device).float()
    grasp_core.load_state_dict(_extract_state_dict(training_args.grasp_ckpt), strict=False)

    C_out = int(model.config.point_backbone_config["backbone_output_dim"])
    T = int(model.config.point_backbone_config["point_token_len"])

    class BBackboneWrapper(nn.Module):
        """
        Outputs:
          hat_f_seq: [B, T, C_out]
          R_loss: scalar
          D_loss: scalar
          feature_loss: scalar
        Also stores last_* for Trainer to read.
        """
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core

            self._logged_path = False

            self.enc_adapter = None
            self._enc_ratio = int(training_args.enc_adapter_ratio)
            self._enc_enabled = bool(training_args.use_enc_adapter)

            self.dec_adapter = None
            if bool(training_args.use_dec_adapter):
                self.dec_adapter = ResidualConv1dAdapter(
                    channels=C_out,
                    ratio=int(training_args.dec_adapter_ratio),
                    use_alpha=True,
                    alpha_init=-4.0,
                ).to(device).float()
                logger.info("✅ Added dec_adapter (ResidualConv1dAdapter, sigmoid-gate).")

            # compatibility (and for distill projection)
            self.adapter1 = nn.Linear(C_out, 768).to(device).float()

            # ---- cached losses ----
            self.last_R_loss = None
            self.last_D_loss = None
            self.last_feature_loss = None

        def _ensure_enc_adapter(self, in_dim: int):
            if self.enc_adapter is not None:
                return
            self.enc_adapter = MinkowskiAdapter(
                channels=in_dim,
                ratio=self._enc_ratio,
                enabled=True,
                use_alpha=True,
            ).to(device).float()
            logger.info(f"✅ enc_adapter init: MinkowskiAdapter(channels={in_dim}, ratio={self._enc_ratio}, sigmoid-gate)")

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

            used_encode_decode = False
            hat_f_seq, R_loss, D_loss = None, None, None

            # prefer encode/decode if exists
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
                    if isinstance(dec_out, (tuple, list)):
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
                # fallback path
                out = self.core.forward(xyz_norm, return_loss=True)
                hat_f_seq, R_loss, D_loss = out

            # log once
            if not self._logged_path:
                logger.info(f"[PATH] GRASP used_encode_decode={used_encode_decode}")
                self._logged_path = True
                try:
                    if self.dec_adapter is not None and getattr(self.dec_adapter, "alpha", None) is not None:
                        gate = torch.sigmoid(self.dec_adapter.alpha.detach()).item()
                        logger.info(
                            f"[SANITY] dec_adapter gate(sigmoid(alpha))={gate:.6f}  alpha={float(self.dec_adapter.alpha.detach().cpu().item()):.6f}"
                        )
                except Exception:
                    pass

            if not torch.is_tensor(hat_f_seq):
                raise RuntimeError("❌ hat_f_seq is not a tensor.")
            if int(hat_f_seq.shape[-1]) != C_out:
                raise RuntimeError(f"❌ GRASP output dim mismatch: got C={int(hat_f_seq.shape[-1])}, expected C_out={C_out}")

            hat_f_seq = self._pad_crop_tokens(hat_f_seq, T)

            if self.dec_adapter is not None:
                hat_f_seq = self.dec_adapter(hat_f_seq)

            # ---- distill feature loss ----
            with torch.no_grad():
                f_teacher = teacher(point_clouds.float())
                f_teacher_g = f_teacher.mean(dim=1) if f_teacher.ndim == 3 else f_teacher

            f_student_g = self.adapter1(hat_f_seq.mean(dim=1))
            feature_loss = F.mse_loss(f_student_g.float(), f_teacher_g.float())

            if R_loss is None:
                R_loss = torch.zeros((), device=point_clouds.device)
            if D_loss is None:
                D_loss = torch.zeros((), device=point_clouds.device)

            # cache for trainer
            self.last_R_loss = R_loss
            self.last_D_loss = D_loss
            self.last_feature_loss = feature_loss

            return hat_f_seq, R_loss, D_loss, feature_loss

    student = BBackboneWrapper(grasp_core).to(device).float()
    model.get_model().point_backbone = student
    logger.info("✅ Injected BBackboneWrapper into model.get_model().point_backbone")


class PointLLMTrainerInternalWithCE(transformers.Trainer):
    """
    total_loss = lambda_ce * lm_ce + distill_alpha * feature + lambda_rec * D + lambda_rate * R
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # HuggingFace will pass: input_ids, attention_mask, labels, point_clouds...
        outputs = model(**inputs)
        lm_ce = outputs.loss  # ✅ CE / LM loss (only counts labels != -100)

        # fetch cached point losses
        pb = getattr(model.get_model(), "point_backbone", None)
        R = getattr(pb, "last_R_loss", None) if pb is not None else None
        D = getattr(pb, "last_D_loss", None) if pb is not None else None
        Fd = getattr(pb, "last_feature_loss", None) if pb is not None else None

        # safe defaults
        dev = lm_ce.device
        if R is None: R = torch.zeros((), device=dev)
        if D is None: D = torch.zeros((), device=dev)
        if Fd is None: Fd = torch.zeros((), device=dev)

        total = (
            float(self.args.lambda_ce) * lm_ce +
            float(model.config.distill_alpha) * Fd +
            float(model.config.lambda_rec) * D +
            float(model.config.lambda_rate) * R
        )

        # log occasionally
        if self.state.global_step % max(1, int(self.args.logging_steps)) == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            try:
                self.log({
                    "loss_total": total.detach().float().item(),
                    "loss_lm_ce": lm_ce.detach().float().item(),
                    "loss_feature": Fd.detach().float().item(),
                    "loss_R": R.detach().float().item(),
                    "loss_D": D.detach().float().item(),
                    "w_lambda_ce": float(self.args.lambda_ce),
                    "w_distill_alpha": float(model.config.distill_alpha),
                    "w_lambda_rec": float(model.config.lambda_rec),
                    "w_lambda_rate": float(model.config.lambda_rate),
                })
            except Exception:
                pass

        return (total, outputs) if return_outputs else total


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger = build_logger(__name__, os.path.join(training_args.output_dir, "train.log"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        local_files_only=bool(LOCAL_FILES_ONLY),
    )
    # keep your special tokens
    tokenizer.add_tokens(["<point_patch>", "<point_start>", "<point_end>"], special_tokens=True)

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = PointLLMLlamaForCausalLMInternal.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=None,
        local_files_only=bool(LOCAL_FILES_ONLY),
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(training_args.device)

    # store weights into config for clarity
    model.config.distill_alpha = float(training_args.distill_alpha)
    model.config.lambda_rec = float(training_args.lambda_rec)
    model.config.lambda_rate = float(training_args.lambda_rate)

    # point backbone cfg
    p_cfg = {
        "point_token_len": int(data_args.point_token_len),
        "backbone_output_dim": 8,
        "point_patch_token_id": tokenizer.convert_tokens_to_ids("<point_patch>"),
    }
    model.config.point_backbone_config = p_cfg
    data_args.point_backbone_config = p_cfg
    model.get_model().point_backbone_config = p_cfg
    try:
        model.get_model().config.point_backbone_config = p_cfg
    except Exception:
        pass

    # inject grasp + adapters
    _inject_internal_adapter_logic(model, training_args, data_args, logger)

    # freeze/unfreeze: only point side
    model.requires_grad_(False)
    active_params = []
    train_keys = ["point_backbone", "point_proj", "pre_proj_adapter"]
    for name, param in model.named_parameters():
        if any(k in name for k in train_keys):
            param.requires_grad = True
            active_params.append(name)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\n" + "=" * 60)
        print("📊 Trainable audit report:")
        print(f"  - Trainable param tensors: {len(active_params)}")
        print("  - includes dec_adapter alpha? ", any("dec_adapter.alpha" in n for n in active_params))
        print(f"  - lambda_ce (LM CE weight): {float(training_args.lambda_ce)}")
        print(f"  - distill_alpha: {float(training_args.distill_alpha)}  lambda_rec: {float(training_args.lambda_rec)}  lambda_rate: {float(training_args.lambda_rate)}")
        print("=" * 60 + "\n")

    data_module = make_object_point_data_module(tokenizer=tokenizer, data_args=data_args)

    # ✅ Use our CE-aware Trainer
    trainer = PointLLMTrainerInternalWithCE(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))

    # save tunables
    tunable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(tunable, os.path.join(training_args.output_dir, "adapter_model.bin"))
    logger.info(f"✅ Done. Saved {len(tunable)} trainable tensors to adapter_model.bin")


if __name__ == "__main__":
    train()
