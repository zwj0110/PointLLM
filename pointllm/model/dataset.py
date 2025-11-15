# pointllm_adapter_train/dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os, hashlib, logging
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("ModelNet40Distill")

# ------------------ utils ------------------

def _seed_from_path(path: str) -> int:
    # 基于路径生成稳定种子，保证每个样本采样可复现
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)

def _list_files_recursively(root: Path, exts=(".ply", ".npy")) -> List[str]:
    """
    递归列出 root 下指定扩展名文件，忽略缓存和隐藏目录/文件。
    忽略目录：.cache_points, .git, __pycache__, 以及任意以 '.' 开头的目录。
    忽略文件：以 '.' 开头的隐藏文件。
    """
    ignore_dirs = {".cache_points", ".git", "__pycache__"}
    out: List[str] = []
    for r, dirnames, filenames in os.walk(root):
        # 过滤目录（原地修改能阻止 os.walk 继续深入）
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in ignore_dirs
        ]
        # 过滤文件
        for f in filenames:
            if f.startswith("."):
                continue
            fl = f.lower()
            if any(fl.endswith(e) for e in exts):
                full = os.path.join(r, f)
                rel  = os.path.relpath(full, root)
                # 双重保险：路径中若包含 .cache_points 也跳过
                if "/.cache_points/" in rel.replace("\\", "/"):
                    continue
                out.append(rel)
    out.sort()
    return out


def _cache_dir_for(root: Path) -> Path:
    d = root / ".cache_points"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _hash_rel(rel: str) -> str:
    return hashlib.md5(rel.encode("utf-8")).hexdigest()[:16]

def _unit_sphere_normalize(pts: np.ndarray) -> np.ndarray:
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts

# ------------------ point loaders ------------------

def _load_points_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expect (N,3) in {path}, got {arr.shape}")
    return arr

def _sample_surface_trimesh(mesh, npoints: int) -> np.ndarray:
    import trimesh
    if npoints and npoints > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, count=npoints)
    else:
        pts = mesh.vertices.copy()
    return np.asarray(pts, dtype=np.float32)

def _load_points_ply(path: str, npoints: int) -> np.ndarray:
    """
    优先按 mesh 读取（有面则在表面均匀采样）；否则按点云读取（取 xyz）。
    """
    import trimesh
    mesh = None
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception:
        mesh = None

    if (mesh is not None and
        hasattr(mesh, "faces") and mesh.faces is not None and len(mesh.faces) > 0):
        return _sample_surface_trimesh(mesh, npoints)

    # 无面或 mesh 失败 → pointcloud 模式
    pc = None
    try:
        pc = trimesh.load(path, force="pointcloud", process=False)
    except Exception as e:
        raise ValueError(f"Failed to read mesh/pointcloud: {path} ({e})")

    if hasattr(pc, "vertices") and pc.vertices is not None and len(pc.vertices) > 0:
        pts = np.asarray(pc.vertices, dtype=np.float32)
    elif hasattr(pc, "points") and pc.points is not None and len(pc.points) > 0:
        pts = np.asarray(pc.points, dtype=np.float32)
    else:
        raise ValueError(f"Pointcloud has no vertices/points: {path}")

    return pts

def _resample_to(pts: np.ndarray, npoints: Optional[int], seed: int) -> np.ndarray:
    if npoints is None or npoints <= 0:
        return pts
    N = pts.shape[0]
    rng = np.random.default_rng(seed)
    if N == npoints:
        return pts
    if N > npoints:
        idx = rng.choice(N, npoints, replace=False)
        return pts[idx]
    # pad
    idx = rng.choice(N, npoints - N, replace=True)
    return np.concatenate([pts, pts[idx]], axis=0)

def _load_points_any(path: str, npoints: Optional[int]) -> np.ndarray:
    if path.lower().endswith(".npy"):
        return _load_points_npy(path)
    if path.lower().endswith(".ply"):
        return _load_points_ply(path, npoints or 0)
    raise ValueError(f"Unsupported extension for {path}")

# ------------------ dataset ------------------

class ModelNet40DistillDataset(Dataset):
    """
    与你的训练脚本保持一致的接口：
      ModelNet40DistillDataset(original_root, compressed_root, split="train", ...)

    假设：
      - original_root 下递归包含 .ply 或 .npy
      - compressed_root 下为“对应的压缩版本”，结构尽量平行
      - 默认严格匹配“相同相对路径 + 相同文件名（含扩展名）”
        若两边扩展名不同，可设 strict_match=False，走“按文件名（忽略目录）匹配”

    返回字段：
      {
        "name": <相对路径（原始端）>,
        "original": FloatTensor (N,3),
        "grasp":    FloatTensor (N,3)
      }
    """
    def __init__(
        self,
        original_root: str,
        compressed_root: str,
        split: str = "train",
        num_points: Optional[int] = None,
        cache_npy: bool = True,
        strict_match: bool = True,          # True: 相同相对路径匹配；False: 按文件名匹配
        normalize_unit_sphere: bool = True, # 零均值 + 单位球
    ):
        super().__init__()
        self.orig_root = Path(original_root).expanduser().resolve()
        self.comp_root = Path(compressed_root).expanduser().resolve()
        self.split = split
        self.num_points = num_points
        self.cache_npy = cache_npy
        self.strict_match = strict_match
        self.normalize_unit_sphere = normalize_unit_sphere

        if not self.orig_root.is_dir():
            raise FileNotFoundError(f"original_root not found: {self.orig_root}")
        if not self.comp_root.is_dir():
            raise FileNotFoundError(f"compressed_root not found: {self.comp_root}")

        orig_rel = _list_files_recursively(self.orig_root, exts=(".ply", ".npy"))
        if not orig_rel:
            raise RuntimeError(f"No .ply/.npy under original_root={self.orig_root}")

        pairs: List[Tuple[str, str, str]] = []  # (rel_orig, abs_orig, abs_comp)

        if self.strict_match:
            missing = []
            for rel in orig_rel:
                o_abs = str(self.orig_root / rel)
                c_abs = str(self.comp_root / rel)
                if not os.path.isfile(c_abs):
                    # 扩展名互换再试：.ply↔.npy
                    base = os.path.splitext(rel)[0]
                    alt_c_abs_npy = str(self.comp_root / f"{base}.npy")
                    alt_c_abs_ply = str(self.comp_root / f"{base}.ply")
                    if os.path.isfile(alt_c_abs_npy):
                        c_abs = alt_c_abs_npy
                    elif os.path.isfile(alt_c_abs_ply):
                        c_abs = alt_c_abs_ply
                    else:
                        missing.append(rel)
                        continue
                pairs.append((rel, o_abs, c_abs))
            if missing:
                raise RuntimeError(
                    f"{len(missing)} compressed files missing with strict_match=True; e.g. {missing[:8]}"
                )
        else:
            # 宽松：按“文件名（含扩展名）”匹配；若失败，再按“忽略扩展名”匹配
            comp_all = _list_files_recursively(self.comp_root, exts=(".ply", ".npy"))
            by_name: Dict[str, List[str]] = {}
            by_stem: Dict[str, List[str]] = {}
            for relc in comp_all:
                name = os.path.basename(relc)
                stem = os.path.splitext(name)[0]
                by_name.setdefault(name, []).append(relc)
                by_stem.setdefault(stem, []).append(relc)

            not_found = []
            for rel in orig_rel:
                o_abs = str(self.orig_root / rel)
                name = os.path.basename(rel)
                stem = os.path.splitext(name)[0]
                cand_rel = None
                if name in by_name and by_name[name]:
                    cand_rel = by_name[name][0]
                elif stem in by_stem and by_stem[stem]:
                    cand_rel = by_stem[stem][0]
                if cand_rel is None:
                    not_found.append(rel)
                    continue
                c_abs = str(self.comp_root / cand_rel)
                pairs.append((rel, o_abs, c_abs))
            if not_found:
                raise RuntimeError(
                    f"{len(not_found)} compressed files not found by relaxed matching; e.g. {not_found[:8]}"
                )

        self.items = pairs
        self.cache_o = _cache_dir_for(self.orig_root)
        self.cache_c = _cache_dir_for(self.comp_root)

        logger.info(f"[DistillDataset] pairs={len(self.items)} | "
                    f"orig={self.orig_root} | comp={self.comp_root} | split={self.split}")

    def __len__(self) -> int:
        return len(self.items)

    def _cache_path(self, cache_root: Path, rel: str, who: str) -> Path:
        """
        who: 'o' or 'c'（original / compressed）
        文件名包含 split、who、点数、是否归一化、rel 的哈希，避免同名冲突
        """
        stem = _hash_rel(rel)
        tag = f"{self.split}_{who}_n{self.num_points or 0}_norm{int(self.normalize_unit_sphere)}_{stem}.npy"
        return cache_root / tag

    def _load_and_prepare(self, abs_path: str, rel: str, who: str) -> np.ndarray:
        cache_root = self.cache_o if who == "o" else self.cache_c
        cp = self._cache_path(cache_root, rel, who)

        if self.cache_npy and cp.is_file():
            pts = np.load(cp)
            # 旧缓存可能未做基本校验
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"Corrupted cache {cp}, got {pts.shape}")
            return pts

        # 读取
        pts = _load_points_any(abs_path, self.num_points)

        # 统一点数（对 .ply 直接载入的情况做采样/补齐；.npy 若本身点数不一，也统一）
        pts = _resample_to(pts, self.num_points, seed=_seed_from_path(abs_path))

        # 归一化
        if self.normalize_unit_sphere:
            pts = _unit_sphere_normalize(pts)

        if self.cache_npy:
            try:
                np.save(cp, pts)
            except Exception as e:
                logger.warning(f"Failed to save cache {cp}: {e}")

        return pts

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel, o_abs, c_abs = self.items[idx]
        pts_o = self._load_and_prepare(o_abs, rel, who="o")  # (N,3)
        pts_c = self._load_and_prepare(c_abs, rel, who="c")  # (N,3)

        # 最终返回 (N,3)，DataLoader 叠成 (B,N,3)
        return {
            "name": rel,
            "original": torch.from_numpy(pts_o).float(),
            "grasp": torch.from_numpy(pts_c).float(),
        }
