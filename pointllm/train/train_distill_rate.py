# -*- coding: utf-8 -*-
"""
Train: Point-level Reconstruction (dominant) + Rate loss
[MODIFIED] Loads config.json directly from local path (PointLLM v1.2).
Keeps all original ReconRateTrainer logic and Sanity Checks.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import transformers
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoConfig
import MinkowskiEngine as ME

DEBUG = int(os.environ.get("POINTLLM_DEBUG", "0"))
LOCAL_FILES_ONLY = int(os.environ.get("LOCAL_FILES_ONLY", "0"))

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pointllm.model.pointllm import PointLLMLlamaForCausalLM
from pointllm.utils import build_logger
from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat


# ---------------------------
# Utilities
# ---------------------------

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


def _rate_from_likelihoods(lk) -> Optional[torch.Tensor]:
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


def _sample_points(x: torch.Tensor, n: int) -> torch.Tensor:
    B, N, C = x.shape
    if N == 0:
        return x.new_zeros((B, n, C))
    if N == n:
        return x
    idx = torch.randint(0, N, (B, n), device=x.device)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))


def chamfer_l2_approx(x: torch.Tensor, y: torch.Tensor, sample_n: int = 2048) -> torch.Tensor:
    x_s = _sample_points(x, sample_n)
    y_s = _sample_points(y, sample_n)
    d = torch.cdist(x_s, y_s, p=2)
    min_xy = d.min(dim=2)[0]
    min_yx = d.min(dim=1)[0]
    return (min_xy.mean(dim=1) + min_yx.mean(dim=1)).mean()


# ---------------------------
# Arguments
# ---------------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    version: Optional[str] = field(default="v1")


@dataclass
class DataArguments:
    data_path: str = field(default="")
    dataset_type: str = field(default="modelnet40")
    use_color: bool = field(default=True)
    pointnum: int = field(default=8192)
    anno_path: str = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=1024)
    grasp_ckpt: Optional[str] = field(default=None)
    grasp_config: Optional[str] = field(default=None)
    lambda_rec: float = field(default=1.0)
    lambda_rate: float = field(default=0.001)
    chamfer_sample_n: int = field(default=2048)
    use_enc_adapter: bool = field(default=True)
    use_dec_adapter: bool = field(default=True)
    enc_adapter_ratio: int = field(default=8)
    dec_adapter_ratio: int = field(default=8)
    voxel_size: float = field(default=0.01)
    fail_on_nondiff: bool = field(default=True)



def _inject_grasp_point_reconstructor(model, training_args, data_args, logger):
    from pointllm.pccai.models.architectures.grasp import GeoResCompression

    if training_args.grasp_ckpt is None or training_args.grasp_config is None:
        raise ValueError("grasp_ckpt / grasp_config must be provided")

    device = training_args.device
    grasp_yaml = yaml.safe_load(open(training_args.grasp_config))
    modules_cfg = _get_modules_cfg(grasp_yaml)
    raw_sd = _extract_state_dict(training_args.grasp_ckpt)

    # [SMART FIX] Align num_points
    target_key = None
    for k in raw_sd.keys():
        if "res_dec" in k and "linear.weight" in k:
            target_key = k
    if target_key:
        weight = raw_sd[target_key]
        out_dim = weight.shape[0]
        implied_num_points = out_dim // 3
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            logger.info(f"🔍 [Auto-Fix] Checkpoint implies num_points={implied_num_points}")
        if 'point_mul' in modules_cfg: modules_cfg['point_mul'] = implied_num_points
        if 'res_dec' in modules_cfg:
            modules_cfg['res_dec']['num_points'] = implied_num_points
            if 'point_mul' in modules_cfg['res_dec']: modules_cfg['res_dec']['point_mul'] = implied_num_points

    # Init Grasp Core
    grasp_core = GeoResCompression(modules_cfg, SimpleNamespace(phase="train")).to(device).float()

    clean_sd = {}
    for k, v in raw_sd.items():
        k_clean = k
        if k_clean.startswith("module."): k_clean = k_clean[7:]
        if k_clean.startswith("pcc_model."): k_clean = k_clean[10:]
        clean_sd[k_clean] = v

    missing, unexpected = grasp_core.load_state_dict(clean_sd, strict=False)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        if len(missing) > 0:
            logger.warning(f"⚠️ GRASP load missing keys (first 5): {missing[:5]}")

    grasp_core = grasp_core.float()

    N = int(data_args.pointnum)
    voxel_size = float(training_args.voxel_size)
    fail_on_nondiff = bool(getattr(training_args, "fail_on_nondiff", True))

    class GraspReconWrapper(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core
            self._logged_once = False

        @staticmethod
        def _points_to_me_coords(xyz: torch.Tensor, voxel_size_: float) -> torch.Tensor:
            B, N_, _ = xyz.shape
            coords_list = []
            for b in range(B):
                q = torch.floor(xyz[b, :, :3] / voxel_size_).to(torch.int32)
                try:
                    uq = ME.utils.sparse_quantize(q)
                except:
                    uq = torch.unique(q, dim=0)
                coords_list.append(uq)
            coords = ME.utils.batched_coordinates(coords_list, dtype=torch.int32)
            return coords

        @staticmethod
        def _resample_to_N(pts_b: torch.Tensor, N_: int, device_: torch.device) -> torch.Tensor:
            m = int(pts_b.shape[0])
            if m == 0: return pts_b.new_zeros((N_, 3))
            if m == N_: return pts_b
            idx = torch.randint(0, m, (N_,), device=device_)
            return pts_b[idx]

        @staticmethod
        def _tensor_info(x):
            if torch.is_tensor(x):
                return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, req_grad={x.requires_grad})"
            return str(type(x))

        @staticmethod
        def _pick_diff_xyz(out: dict):
            if not isinstance(out, dict):
                return None, None
            candidates = [
                "xyz_hat", "x_rec", "x_recon", "x_hat_float", "points_hat",
                "recon_points", "decoded_xyz", "x_out"
            ]
            for k in candidates:
                v = out.get(k, None)
                if torch.is_tensor(v) and v.dim() == 3 and v.size(-1) == 3 and v.requires_grad:
                    return k, v
            return None, None

        def forward(self, point_clouds: torch.Tensor):
            xyz_norm = _normalize_unit_sphere(point_clouds)
            B = int(point_clouds.shape[0])

            with torch.cuda.amp.autocast(enabled=False):
                coords_int = self._points_to_me_coords(xyz_norm, voxel_size).to(point_clouds.device).int()
                out = self.core.forward(coords_int)

                if (not self._logged_once) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    logger.info("=" * 80)
                    logger.info("[SANITY] GRASP out keys & tensor infos")
                    if isinstance(out, dict):
                        for k in sorted(list(out.keys())):
                            v = out[k]
                            if torch.is_tensor(v):
                                logger.info(f"  - {k}: {self._tensor_info(v)}")
                            elif isinstance(v, dict):
                                logger.info(f"  - {k}: dict(keys={list(v.keys())[:50]})")
                            else:
                                logger.info(f"  - {k}: {type(v)}")

                    for enc_name in ["vox_enc", "res_enc", "sparse_enc", "enc", "encoder"]:
                        ve = getattr(self.core, enc_name, None)
                        if hasattr(ve, "_calls"):
                            logger.info(f"[ENC_CALLS] {enc_name} calls={ve._calls}, adapter_calls={ve._adapter_calls}")
                            break
                    logger.info("=" * 80)

                R_loss = None
                if isinstance(out, dict):
                    lk = out.get("likelihoods", None)
                    R_loss = _rate_from_likelihoods(lk)
                if R_loss is None: R_loss = torch.zeros((), device=point_clouds.device)

                name, xyz_diff = self._pick_diff_xyz(out)
                if xyz_diff is not None:
                    counts = []
                    seq = []
                    for b in range(B):
                        pts_b = xyz_diff[b]
                        counts.append(int(pts_b.shape[0]))
                        seq.append(self._resample_to_N(pts_b, N_=N, device_=point_clouds.device).unsqueeze(0))
                    xyz_hat = torch.cat(seq, dim=0)

                    if (not self._logged_once) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                        logger.info(f"✅ [RECON] Using differentiable recon out['{name}']")
                        self._logged_once = True
                    return xyz_hat, R_loss, counts

                # Fallback: use out['x_hat']
                if not (isinstance(out, dict) and "x_hat" in out and torch.is_tensor(out["x_hat"])):
                    raise RuntimeError("❌ No differentiable recon found.")

                x_hat = out["x_hat"]
                if torch.is_floating_point(x_hat[:, 0]):
                    b_idx = x_hat[:, 0].round().long()
                else:
                    b_idx = x_hat[:, 0].long()

                if (not x_hat.requires_grad) and fail_on_nondiff:
                    raise RuntimeError("⚠️ [NONDIFF] x_hat has no grad! Cannot train.")

                xyz_all = x_hat[:, 1:4].float() * voxel_size
                seq_list, counts = [], []
                for b in range(B):
                    pts_b = xyz_all[b_idx == b]
                    counts.append(int(pts_b.shape[0]))
                    pts_b = self._resample_to_N(pts_b, N_=N, device_=point_clouds.device)
                    seq_list.append(pts_b.unsqueeze(0))

                xyz_hat = torch.cat(seq_list, dim=0)

                if not self._logged_once and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    logger.info(f"[DECODE] xyz_hat shape={tuple(xyz_hat.shape)} req_grad={xyz_hat.requires_grad}")
                    self._logged_once = True

                return xyz_hat, R_loss, counts

    wrapper = GraspReconWrapper(grasp_core).to(device).float()
    model.get_model().point_backbone = wrapper
    logger.info("✅ Injected GraspReconWrapper into model.get_model().point_backbone")


# ---------------------------
# Trainer
# ---------------------------

class ReconRateTrainer(transformers.Trainer):
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._user_logger = logger
        self._reqgrad_logged = False
        # [MODIFIED] 1. 初始化一个字典来暂存分项 loss
        self._custom_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        point_clouds = inputs.get("point_clouds", None)
        pb = getattr(model.get_model(), "point_backbone", None)

        with torch.cuda.amp.autocast(enabled=False):
            xyz_hat, R_loss, counts = pb(point_clouds.float())

        xyz_gt = _normalize_unit_sphere(point_clouds.float())
        sample_n = getattr(self.args, "chamfer_sample_n", 2048)
        rec_loss = chamfer_l2_approx(xyz_hat, xyz_gt, sample_n=sample_n)

        lam_rec = float(model.config.lambda_rec)
        lam_rate = float(model.config.lambda_rate)
        total = lam_rec * rec_loss + lam_rate * R_loss
        if model.training:
            self._custom_metrics = {
                "rec_loss": round(rec_loss.detach().item(), 6),
                "rate_loss": round(R_loss.detach().item(), 6)
            }

        # Log grads once
        if (not self._reqgrad_logged) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            if self._user_logger:
                self._user_logger.info(f"[REQ_GRAD] xyz_hat.requires_grad={xyz_hat.requires_grad}")
            self._reqgrad_logged = True

        return (total, {"rec_loss": rec_loss, "R_loss": R_loss}) if return_outputs else total


    def log(self, logs: Dict[str, float]) -> None:
        if self._custom_metrics:
            logs.update(self._custom_metrics)
        super().log(logs)


def data_collator_recon(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    pcs = [b["point_clouds"] for b in batch]
    pcs = torch.stack(pcs, dim=0)
    return {"point_clouds": pcs}


# ---------------------------
# Callbacks
# ---------------------------

class AdapterGradSanityCallback(transformers.TrainerCallback):
    def __init__(self, logger, only_rank0=True, max_print=50):
        self.logger = logger
        self.only_rank0 = only_rank0
        self.max_print = max_print
        self._done = False

    def on_backward_end(self, args, state, control, **kwargs):
        # 只在第一次反向传播时运行
        if self._done: return
        if self.only_rank0 and int(os.environ.get("LOCAL_RANK", "0")) != 0: return

        model = kwargs.get("model")
        enc_grads = []
        dec_grads = []

        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            # 只要名字里带 adapter 就抓出来检查
            if "adapter" in n:
                g = p.grad
                # 计算梯度的绝对值平均数
                val = float(g.detach().abs().mean().cpu().item()) if g is not None else None

                if "vox_dec" in n or "res_dec" in n:
                    dec_grads.append((n, val))
                else:
                    enc_grads.append((n, val))

        self.logger.info("=" * 60)
        self.logger.info("[GRAD_SANITY] Adapter Gradient Check (mean|grad|)")

        self.logger.info("--- Decoder Gradients (关键看这里) ---")
        if not dec_grads:
            self.logger.warning("⚠️ No Decoder parameters found with requires_grad=True!")
        for n, v in dec_grads:
            status = "✅" if (v is not None and v > 0) else "❌ ZERO/NONE"
            self.logger.info(f"  {status} {n}: {v}")

        self.logger.info("--- Encoder Gradients (Top 5) ---")
        for n, v in enc_grads[:5]:
            self.logger.info(f"  - {n}: {v}")

        self.logger.info("=" * 60)
        self._done = True


class AdapterUpdateSanityCallback(transformers.TrainerCallback):
    def __init__(self, logger, only_rank0=True):
        self.logger = logger
        self.only_rank0 = only_rank0
        self._snap = {}
        self._done = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.only_rank0 and int(os.environ.get("LOCAL_RANK", "0")) != 0: return
        model = kwargs.get("model")
        # 记录所有 trainable adapter 的初始状态
        for n, p in model.named_parameters():
            if p.requires_grad and "adapter" in n:
                self._snap[n] = p.detach().float().cpu().clone()

    def on_step_end(self, args, state, control, **kwargs):
        # 只在第一步结束后运行
        if self._done: return
        if self.only_rank0 and int(os.environ.get("LOCAL_RANK", "0")) != 0: return

        model = kwargs.get("model")
        diffs_dec = []
        diffs_enc = []

        for n, p in model.named_parameters():
            if n in self._snap:
                # 计算参数变化的绝对值
                d = (p.detach().float().cpu() - self._snap[n]).abs().mean().item()

                if "vox_dec" in n or "res_dec" in n:
                    diffs_dec.append((d, n))
                else:
                    diffs_enc.append((d, n))

        # 降序排列，变化最大的在前面
        diffs_dec.sort(reverse=True)
        diffs_enc.sort(reverse=True)

        self.logger.info("=" * 60)
        self.logger.info("[UPDATE_SANITY] mean|Δparam| after step 1")

        self.logger.info("--- Decoder Updates (关键看这里) ---")
        if not diffs_dec:
            self.logger.warning("⚠️ No Decoder parameters were tracked!")
        for d, n in diffs_dec:
            status = "✅" if d > 0 else "❌ ZERO"
            self.logger.info(f"  {status} {n}: {d:.6e}")

        self.logger.info("--- Encoder Updates (Top 5) ---")
        for d, n in diffs_enc[:5]:
            self.logger.info(f"  - {n}: {d:.6e}")

        self.logger.info("=" * 60)
        self._done = True


def _dump_core_modules(model, logger):
    if int(os.environ.get("LOCAL_RANK", "0")) != 0: return
    logger.info("=" * 80)
    logger.info("[CORE_DUMP] listing modules under model.point_backbone.core ...")
    core = getattr(getattr(model.get_model(), "point_backbone", None), "core", None)
    if core:
        for n, m in core.named_modules():
            logger.info(f"[CORE_MOD] {n} :: {m.__class__.__name__}")
    logger.info("=" * 80)


def _register_decoder_forward_hooks(model, logger):
    if int(os.environ.get("LOCAL_RANK", "0")) != 0: return
    hooked = []
    for n, m in model.named_modules():
        if "point_backbone.core.vox_dec.adapter" in n:
            m.register_forward_hook(lambda m, i, o: logger.debug(f"[DEC_FWD] called."))
            hooked.append(n)
    logger.info(f"[HOOK] registered decoder forward hooks: {len(hooked)}")


def _optimizer_sanity(trainer, model, logger):
    if int(os.environ.get("LOCAL_RANK", "0")) != 0: return
    trainer.create_optimizer()
    logger.info(f"[OPT_SANITY] param_groups={len(trainer.optimizer.param_groups)}")


# ---------------------------
# Train entry
# ---------------------------

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)
    logger = build_logger(__name__, os.path.join(training_args.output_dir, "train.log"))

    # =========================================================================
    # [FIX] Load Config & Tokenizer from Local Path (PointLLM v1.2)
    # =========================================================================
    logger.info(f"Loading Config from: {model_args.model_name_or_path}")
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    training_args.save_strategy = "no"
    training_args.save_total_limit = 1
    config.lambda_rec = float(training_args.lambda_rec)
    config.lambda_rate = float(training_args.lambda_rate)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False, trust_remote_code=True
    )
    tokenizer.add_tokens(["<point_patch>", "<point_start>", "<point_end>"], special_tokens=True)

    # Load Model (Using the local config)
    if training_args.bf16:
        llm_dtype = torch.bfloat16
    elif training_args.fp16:
        llm_dtype = torch.float16
    else:
        llm_dtype = torch.float16

    logger.info(f"Loading Model from {model_args.model_name_or_path}...")
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,  # Direct load
        torch_dtype=llm_dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True
    )

    # [CRITICAL] Inject Float32 Hooks immediately after loading
    # This is required for 4090 to avoid Type Mismatch in PointBERT/GraspNet
    if hasattr(model.get_model(), "point_backbone"):
        logger.info("🔧 Injecting Float32 Hooks for PointBackbone...")
        model.get_model().point_backbone.float()

        def cast_output_to_half(module, input, output):
            if isinstance(output, torch.Tensor):
                return output.to(llm_dtype)
            return output

        model.get_model().point_backbone.register_forward_hook(cast_output_to_half)

    # =========================================================================

    _inject_grasp_point_reconstructor(model, training_args, data_args, logger)

    train_dataset = ModelNet40DirCompat(
        root=data_args.data_path,
        split="train",
        npoints=int(data_args.pointnum),
        use_color=bool(data_args.use_color),
    )

    # Sanity: Run one dry forward
    try:
        model.eval()
        with torch.no_grad():
            tmp = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)))
            _ = model.get_model().point_backbone(tmp["point_clouds"].to(training_args.device).float())
        logger.info("✅ Ran one dry forward (SANITY printed once).")
    except Exception as e:
        logger.warning(f"Dry forward failed (still OK): {e}")
    finally:
        model.train()

    # Trainable params setup
    use_enc = bool(training_args.use_enc_adapter)
    use_dec = bool(training_args.use_dec_adapter)


    model.requires_grad_(False)
    trainable = []
    for n, m in model.named_modules():
        # 只要类名是 MinkowskiAdapter，就认为是我们要训练的目标
        # (这样兼容了 adapters.py 和 autoencoder.py 里定义的两个不同来源的同名类)
        if m.__class__.__name__ == "MinkowskiAdapter":

            # 判断是 Encoder 还是 Decoder
            # 注意：res_dec (Residual Decoder) 也应该算作 Decoder
            is_dec_module = ("vox_dec" in n) or ("res_dec" in n)
            is_enc_module = not is_dec_module

            if is_dec_module and not use_dec: continue
            if is_enc_module and not use_enc: continue

            # 激活该模块下的所有参数
            for pn, p in m.named_parameters():
                p.requires_grad = True
                full_param_name = f"{n}.{pn}"
                trainable.append(full_param_name)

        # ==============================================================================
        # [VERIFY] 打印检查 (现在你应该能看到所有 adapter 了)
        # ==============================================================================
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print("\n" + "!" * 80)
        print(f"🔥 TRAINABLE PARAMETERS VERIFICATION ({len(trainable)} tensors) 🔥")
        print("!" * 80)

        enc_list = [t for t in trainable if "vox_enc" in t or "res_enc" in t]
        dec_list = [t for t in trainable if "vox_dec" in t or "res_dec" in t]

        print(f"\n--- Encoder Params ({len(enc_list)}) ---")
        # 只打印前5个和后5个，防止刷屏，但你可以全打印
        for t in enc_list[:5]: print(f"  ✅ {t}")
        if len(enc_list) > 10: print("  ... (omitted) ...")

        print(f"\n--- Decoder Params ({len(dec_list)}) ---")
        for t in dec_list[:5]: print(f"  ✅ {t}")
        if len(dec_list) > 10: print("  ... (omitted) ...")

        # 再次确认：必须要有参数！
        if len(trainable) == 0:
            raise RuntimeError("❌ NO TRAINABLE PARAMETERS FOUND! Adapter activation failed.")
        print("!" * 80 + "\n")

    _dump_core_modules(model, logger)
    _register_decoder_forward_hooks(model, logger)

    trainer = ReconRateTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator_recon,
        callbacks=[
            AdapterGradSanityCallback(logger=logger),
            AdapterUpdateSanityCallback(logger=logger),
        ],
        logger=logger,
    )

    _optimizer_sanity(trainer, model, logger)

    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))

    tunable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(tunable, os.path.join(training_args.output_dir, "adapter_model.bin"))
    logger.info(f"✅ Done. Saved {len(tunable)} trainable tensors to adapter_model.bin")


if __name__ == "__main__":
    train()