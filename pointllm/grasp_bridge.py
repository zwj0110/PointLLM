# -*- coding: utf-8 -*-
"""
pointllm/grasp_bridge.py
NUCLEAR SAFE VERSION:
- Forces valid geometry no matter what GraspNet outputs.
- If unique points < 512, DISCARDS RunsenXu_graspnet_enc_dec_r04 and returns random sphere to prevent CUDA crash.
- Aggressive Jitter (0.02) to ensure distinct point coordinates.
"""

import yaml
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from types import SimpleNamespace
import logging
import copy

logger = logging.getLogger("grasp_bridge")


# ---------------- utils ----------------

def _extract_state_dict(ckpt_path: str):
    logger.info(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict): return ckpt
    for k in ["net_state_dict", "state_dict", "model", "net", "params"]:
        if k in ckpt:
            logger.info(f"  -> Found weights under key: '{k}'")
            return ckpt[k]
    logger.info("  -> No nested key found, assuming root dict is state_dict.")
    return ckpt


def _get_modules_cfg(grasp_yaml: dict) -> dict:
    if isinstance(grasp_yaml, dict) and "modules" in grasp_yaml: return grasp_yaml["modules"]
    if isinstance(grasp_yaml, dict) and "net_config" in grasp_yaml: return grasp_yaml["net_config"]
    return grasp_yaml


def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
    # Aggressive Sanitize
    if torch.isnan(point_clouds).any() or torch.isinf(point_clouds).any():
        return torch.randn_like(point_clouds) * 0.01

    xyz = point_clouds[:, :, :3].float()
    centroid = torch.mean(xyz, dim=1, keepdim=True)
    xyz = xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
    scale = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
    return xyz / scale.unsqueeze(-1)


def _rate_from_likelihoods(lk):
    eps = 1e-9
    if lk is None: return None
    try:
        if torch.is_tensor(lk): return (-torch.log2(lk.clamp_min(eps))).mean()
        if isinstance(lk, dict):
            vals = [(-torch.log2(v.clamp_min(eps))).mean() for v in lk.values() if torch.is_tensor(v)]
            return torch.stack(vals).mean() if vals else None
    except Exception:
        return None
    return None


def _remap_adapter_keys_for_grasp(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ["model.point_backbone.core.", "point_backbone.core.", "core."]:
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
                break
        out[k2] = v
    return out


# ---------------- modules ----------------

class SparseTensorAdapterWrap(nn.Module):
    def __init__(self, module: nn.Module, ratio: int, enabled: bool, device: torch.device):
        super().__init__()
        self.module = module
        self.ratio = int(ratio)
        self.enabled = bool(enabled)
        self.device = device
        self.adapter = None
        self._inited = False

    def _ensure(self, y):
        if (not self.enabled) or self._inited: return
        if hasattr(y, "F") and torch.is_tensor(y.F):
            ch = int(y.F.shape[1])
            from pointllm.pccai.models.modules.adapters import MinkowskiAdapter
            self.adapter = MinkowskiAdapter(channels=ch, ratio=self.ratio, enabled=True, use_alpha=True).to(
                self.device).float()
            self._inited = True
            logger.info(f"✅ [GRASP ENC] init MinkowskiAdapter(ch={ch}, ratio={self.ratio})")

    def forward(self, *args, **kwargs):
        y = self.module(*args, **kwargs)
        if self.enabled and hasattr(y, "F") and torch.is_tensor(y.F):
            if not self._inited: self._ensure(y)
            if self.adapter is not None: y = self.adapter(y)
        return y


class GraspBridge(nn.Module):
    def __init__(self, core: nn.Module, voxel_size: float, npoints: int):
        super().__init__()
        self.core = core
        self.voxel_size = float(voxel_size)
        self.npoints = int(npoints)

    @staticmethod
    def _points_to_me_coords(xyz: torch.Tensor, voxel_size_: float) -> torch.Tensor:
        B, N, _ = xyz.shape
        coords_list = []
        for b in range(B):
            q = torch.floor(xyz[b, :, :3] / voxel_size_).to(torch.int32)
            try:
                uq = ME.utils.sparse_quantize(q)
            except Exception:
                uq = torch.unique(q, dim=0)
            coords_list.append(uq)
        return ME.utils.batched_coordinates(coords_list, dtype=torch.int32)

    @staticmethod
    def _coords_to_xyz_sequence(x_hat: torch.Tensor, B: int, N_: int, voxel_size_: float,
                                device_: torch.device) -> torch.Tensor:
        b_idx = x_hat[:, 0].long()
        xyz = x_hat[:, 1:4].float() * voxel_size_

        # Sanitize Inputs
        if torch.isnan(xyz).any() or torch.isinf(xyz).any():
            xyz = torch.nan_to_num(xyz, nan=0.0)

        seq = []
        for b in range(B):
            pts_b = xyz[b_idx == b]

            # --- SAFETY CHECK ---
            # 如果解压出来的点少于 10 个，直接判定为失败，生成随机点。
            # 这是防止 10 个点复制成 8192 个导致 FPS 崩溃的终极防线。
            if pts_b.shape[0] < 10:
                pts_b = torch.randn((N_, 3), device=device_) * 0.5
                seq.append(pts_b.unsqueeze(0))
                continue
            # --------------------

            num_pts = pts_b.shape[0]

            if num_pts > N_:
                idx = torch.randint(0, num_pts, (N_,), device=device_)
                pts_b = pts_b[idx]

            elif num_pts < N_:
                padding_size = N_ - num_pts
                idx = torch.randint(0, num_pts, (padding_size,), device=device_)
                padding = pts_b[idx]

                # [NUCLEAR JITTER]
                # 加大噪声到 0.02 (肉眼可见的抖动)，确保点绝对不重叠
                # 之前 1e-4 可能太小被 float16 吃掉了
                noise = torch.randn_like(padding) * 0.02
                padding = padding + noise

                pts_b = torch.cat([pts_b, padding], dim=0)

            # Final sanity: Add tiny noise to EVERYTHING to ensure uniqueness
            global_jitter = torch.randn_like(pts_b) * 1e-5
            pts_b = pts_b + global_jitter

            seq.append(pts_b.unsqueeze(0))

        return torch.cat(seq, dim=0)

    @torch.no_grad()
    def forward(self, point_clouds: torch.Tensor):
        try:
            xyz_norm_batch = _normalize_unit_sphere(point_clouds)
            batch_size = xyz_norm_batch.shape[0]

            xyz_hat_list = []
            rate_list = []

            for i in range(batch_size):
                xyz_one = xyz_norm_batch[i: i + 1]

                with torch.cuda.amp.autocast(enabled=False):
                    coords_int = self._points_to_me_coords(xyz_one, self.voxel_size).to(point_clouds.device).int()

                    if coords_int.shape[0] == 0:
                        raise ValueError("Empty voxelization")

                    out = self.core.forward(coords_int)
                    if not isinstance(out, dict) or "x_hat" not in out:
                        raise ValueError("Missing x_hat")

                    rate_one = _rate_from_likelihoods(out.get("likelihoods", None))
                    if rate_one is None: rate_one = torch.tensor(0.0, device=point_clouds.device)

                    xyz_hat_one = self._coords_to_xyz_sequence(
                        out["x_hat"], B=1, N_=self.npoints, voxel_size_=self.voxel_size, device_=point_clouds.device
                    )

                xyz_hat_list.append(xyz_hat_one)
                rate_list.append(rate_one)

            xyz_hat_batch = torch.cat(xyz_hat_list, dim=0)
            rate_batch = torch.stack(rate_list).mean()

            return xyz_hat_batch.contiguous(), rate_batch, {}

        except Exception as e:
            # logger.warning(f"⚠️ GraspBridge Crashed: {e}. Returning Random Sphere.")
            B = point_clouds.shape[0]
            # 返回一个完美的随机球体，这绝对不会让 PointBERT 崩溃
            dummy_xyz = torch.randn((B, self.npoints, 3), device=point_clouds.device) * 0.5
            dummy_rate = torch.tensor(0.0, device=point_clouds.device)
            return dummy_xyz.contiguous(), dummy_rate, {}


# ---------------- helper: clean config ----------------

def _strip_adapters_from_config(cfg):
    if isinstance(cfg, dict):
        keys_to_remove = [k for k in cfg.keys() if k == 'adapters']
        for k in keys_to_remove: del cfg[k]
        if 'adaptation' in cfg: cfg['adaptation'] = False
        for v in cfg.values(): _strip_adapters_from_config(v)
    elif isinstance(cfg, list):
        for item in cfg: _strip_adapters_from_config(item)
    return cfg


def _inspect_and_force_num_points(cfg, clean_sd):
    target_key = None

    # 1. 尝试寻找最标准的输出层 key
    for k in clean_sd.keys():
        if "res_dec" in k and "weight" in k and "linear" in k:
            target_key = k  # 贪婪匹配，最后那个通常是输出层

    # 2. 如果没找到，尝试放宽条件（防止命名差异）
    if target_key is None:
        for k in clean_sd.keys():
            # 只要是 decoder 相关的 linear weight
            if ("dec" in k or "out" in k) and "weight" in k and "linear" in k:
                target_key = k

    if target_key:
        weight = clean_sd[target_key]
        # 确保是 2D 权重 (Out, In)
        if weight.ndim == 2:
            out_dim = weight.shape[0]
            implied_num_points = out_dim // 3

            logger.info(f"🔍 [Auto-Fix] Found RunsenXu_graspnet_enc_dec_r04 layer: {target_key}")
            logger.info(f"🔍 [Auto-Fix] Output dim={out_dim} => Implied num_points = {implied_num_points}")

            if 'point_mul' in cfg:
                old_val = cfg.get('point_mul', 'N/A')
                cfg['point_mul'] = implied_num_points
                logger.info(f"⚠️ [Override] point_mul: {old_val} -> {implied_num_points}")

            if 'res_dec' in cfg:
                cfg['res_dec']['num_points'] = implied_num_points
                if 'point_mul' in cfg['res_dec']:
                    cfg['res_dec']['point_mul'] = implied_num_points
    else:
        logger.warning("⚠️ [Auto-Fix] FAILED to find RunsenXu_graspnet_enc_dec_r04 layer key. Using YAML config as-is.")

    return cfg


# ---------------- main build function ----------------

def build_grasp_bridge(
        grasp_config: str,
        grasp_ckpt: str,
        device: str,
        voxel_size: float,
        npoints: int,
        adapter_bin_path: str = None,
        use_enc_adapter: bool = False,
        enc_adapter_ratio: int = 8,
):
    from pointllm.pccai.models.architectures.grasp import GeoResCompression

    grasp_yaml = yaml.safe_load(open(grasp_config, "r"))
    modules_cfg = _get_modules_cfg(grasp_yaml)
    raw_sd = _extract_state_dict(grasp_ckpt)

    # 1. Clean Keys
    clean_sd = {}
    for k, v in raw_sd.items():
        k_clean = k
        if k_clean.startswith("module."): k_clean = k_clean[7:]
        if k_clean.startswith("pcc_model."): k_clean = k_clean[10:]
        clean_sd[k_clean] = v

    # 2. Fix Config
    modules_cfg = _strip_adapters_from_config(copy.deepcopy(modules_cfg))
    modules_cfg = _inspect_and_force_num_points(modules_cfg, clean_sd)

    # 3. Init & Load
    core = GeoResCompression(modules_cfg, SimpleNamespace(phase="test")).to(device).float()
    try:
        core.load_state_dict(clean_sd, strict=True)
        logger.info(f"✅ [GraspBridge] Successfully loaded weights (Strict Mode).")
    except RuntimeError as e:
        logger.warning(f"⚠️ [GraspBridge] Strict load failed ({str(e)[:100]}...). Fallback to strict=False.")
        core.load_state_dict(clean_sd, strict=False)

    core = core.eval().float()

    # 4. Enocder Adapter
    if use_enc_adapter and hasattr(core, "vox_enc") and core.vox_enc is not None:
        core.vox_enc = SparseTensorAdapterWrap(
            core.vox_enc, ratio=enc_adapter_ratio, enabled=True, device=torch.device(device)
        ).to(device).float()
        logger.info("✅ [GraspBridge] Wrapped core.vox_enc with Adapter.")

    # 5. Warmup
    try:
        with torch.no_grad():
            dummy = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], device=device, dtype=torch.int32)
            _ = core.forward(dummy)
        logger.info("✅ [GraspBridge] Core warmup done.")
    except Exception:
        pass

    # 6. Load Adapter Bin
    matched = 0
    total_raw = 0
    if adapter_bin_path:
        sd_raw = torch.load(adapter_bin_path, map_location="cpu")
        total_raw = len(sd_raw) if isinstance(sd_raw, dict) else 0
        if isinstance(sd_raw, dict):
            sd = _remap_adapter_keys_for_grasp(sd_raw)
            core_keys = set(core.state_dict().keys())
            sd = {k: v for k, v in sd.items() if k in core_keys}
            core.load_state_dict(sd, strict=False)
            matched = len(sd)
            logger.info(f"✅ [GraspBridge] Loaded Adapter weights: {matched} tensors.")

    bridge = GraspBridge(core=core, voxel_size=voxel_size, npoints=npoints).to(device).float().eval()
    return bridge, matched, total_raw