# pointllm/data/modelnet_dir_dataset.py
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
import hashlib
import logging

logger = logging.getLogger("ModelNet40Dir")

def _seed_from_path(path: str) -> int:
    # 根据文件路径生成固定种子，确保每个样本每次采样一致（可复现）
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)

def _read_shape_names(root: Path) -> Optional[List[str]]:
    candidates = [
        root / "modelnet40_shape_names.txt",
        root / "ModelNet40" / "modelnet40_shape_names.txt",
    ]
    for p in candidates:
        if p.is_file():
            cats = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            return cats
    return None

def _list_categories(root: Path) -> List[str]:
    cats = _read_shape_names(root)
    if cats:
        return cats
    scan_root = root / "ModelNet40" if (root / "ModelNet40").is_dir() else root
    dirs = []
    for d in sorted(scan_root.iterdir()):
        if d.is_dir():
            has_geom = any(
                f.suffix.lower() in (".off", ".ply")
                for f in d.rglob("*")
                if f.is_file()
            )
            if has_geom:
                dirs.append(d.name)
    if not dirs:
        raise FileNotFoundError(f"No categories found under {scan_root}.")
    return dirs

def _try_read_filelist(root: Path, split: str) -> Optional[List[str]]:
    candidates = [
        root / f"modelnet40_{split}.txt",
        root / "ModelNet40" / f"modelnet40_{split}.txt",
    ]
    for p in candidates:
        if p.is_file():
            items = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            return items
    return None

def _collect_files_from_dirs(root: Path, categories: List[str], split: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    base = root / "ModelNet40" if (root / "ModelNet40").is_dir() else root
    for cat in categories:
        cat_dir = base / cat
        if not cat_dir.is_dir():
            continue
        split_dir = cat_dir / split
        if split_dir.is_dir():
            exts = ("*.off", "*.ply", "*.OFF", "*.PLY")
            for ext in exts:
                for f in sorted(split_dir.glob(ext)):
                    if f.is_file():
                        pairs.append((str(f), cat))
        else:
            for ext in ("*.off", "*.ply", "*.OFF", "*.PLY"):
                for f in sorted(cat_dir.glob(ext)):
                    if f.is_file():
                        pairs.append((str(f), cat))
    if not pairs:
        raise FileNotFoundError(f"No files found for split='{split}'.")
    return pairs

def _resolve_from_filelist(root: Path, filelist: List[str], categories: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    base = root / "ModelNet40" if (root / "ModelNet40").is_dir() else root
    for rel in filelist:
        rel = rel.strip()
        cand = base / rel
        if not cand.suffix:
            off = cand.with_suffix(".off")
            ply = cand.with_suffix(".ply")
            if off.is_file():
                cand = off
            elif ply.is_file():
                cand = ply
        if not cand.is_file():
            parts = rel.split("/")
            if len(parts) == 2:
                maybe = base / parts[0] / parts[1]
                if maybe.is_file():
                    cand = maybe
        if not cand.is_file():
            raise FileNotFoundError(f"List item not found: {rel} under {base}")
        cat = cand.parent.name
        if cat not in categories:
            gp = cand.parent.parent.name if cand.parent.parent else None
            if gp in categories:
                cat = gp
            else:
                cand_cat = rel.split("/")[0]
                if cand_cat in categories:
                    cat = cand_cat
                else:
                    raise ValueError(f"Cannot resolve category for: {rel}")
        pairs.append((str(cand), cat))
    return pairs

def _sample_points_from_mesh(mesh: "trimesh.Trimesh", npoints: int) -> np.ndarray:
    if npoints and npoints > 0:
        pts, _ = trimesh.sample.sample_surface(mesh, count=npoints)
    else:
        pts = mesh.vertices.copy()
    if not isinstance(pts, np.ndarray):
        pts = np.asarray(pts)
    return pts.astype(np.float32)

# def _load_points(path: str, npoints: int) -> np.ndarray:
#     sfx = Path(path).suffix.lower()
#     if sfx in (".off", ".ply"):
#         mesh = trimesh.load(path, force="mesh", process=False)
#     else:
#         raise ValueError(f"Unsupported file type: {sfx} for {path}")
#     if mesh is None or (hasattr(mesh, "vertices") and len(mesh.vertices) == 0):
#         raise ValueError(f"Failed to load mesh or mesh is empty: {path}")
#     return _sample_points_from_mesh(mesh, npoints)

def _load_points(path: str, npoints: int) -> np.ndarray:
    sfx = Path(path).suffix.lower()
    if sfx not in (".off", ".ply"):
        raise ValueError(f"Unsupported file type: {sfx} for {path}")

    mesh = None
    # 1) 尝试按 mesh 读取（若有面则可在表面采样）
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception:
        mesh = None

    mesh_has_faces = (
        mesh is not None
        and hasattr(mesh, "faces")
        and mesh.faces is not None
        and len(mesh.faces) > 0
    )

    if mesh_has_faces:
        # 在三角面上均匀采样 npoints
        pts = _sample_points_from_mesh(mesh, npoints)
        return pts.astype(np.float32)

    # 2) mesh 为空或无面：按 pointcloud 读取（只取 xyz）
    try:
        pc = trimesh.load(path, force="pointcloud", process=False)
    except Exception as e:
        raise ValueError(f"Failed to load mesh or pointcloud: {path} ({e})")

    if hasattr(pc, "vertices") and pc.vertices is not None and len(pc.vertices) > 0:
        pts = np.asarray(pc.vertices, dtype=np.float32)
    elif hasattr(pc, "points") and pc.points is not None and len(pc.points) > 0:
        pts = np.asarray(pc.points, dtype=np.float32)
    else:
        raise ValueError(f"Pointcloud has no vertices/points: {path}")

    # 重采样/补齐到 npoints（使用文件路径派生的固定种子→可复现）
    N = pts.shape[0]
    if N == 0:
        raise ValueError(f"Empty point set: {path}")
    if npoints and npoints > 0:
        rng = np.random.default_rng(_seed_from_path(path))
        if N >= npoints:
            idx = rng.choice(N, npoints, replace=False)
        else:
            idx = rng.choice(N, npoints, replace=True)
        pts = pts[idx]

    return pts.astype(np.float32)


class ModelNet40Dir(Dataset):
    """
    返回：
      {'points': Tensor(N,3), 'label': int, 'path': str}
    归一化：零均值 + 单位球（与很多 ModelNet40 流程一致）。
    其它特征（高度/伪颜色）由兼容层追加，保证和 .dat 读取的行为对齐。
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        npoints: int = 8192,
        cache_npy: bool = True,
        subset_nums: int = -1,
        add_height: bool = False,      # 仅用于缓存区分文件名；真正通道拼接在 compat 里做
        gravity_dim: int = 1,          # 默认 y 轴为重力方向（和常见设置一致）
    ) -> None:
        super().__init__()
        split = split.lower()
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.npoints = int(npoints) if npoints is not None else 0
        self.cache_npy = cache_npy
        self.add_height = add_height
        self.gravity_dim = int(gravity_dim)

        categories = _list_categories(self.root)
        self.categories = sorted(categories)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.categories)}

        filelist = _try_read_filelist(self.root, split)
        if filelist:
            pairs = _resolve_from_filelist(self.root, filelist, self.categories)
        else:
            pairs = _collect_files_from_dirs(self.root, self.categories, split)

        if subset_nums is not None and subset_nums > 0:
            pairs = pairs[:subset_nums]

        self.samples: List[Tuple[str, int]] = [(p, self.class_to_idx[c]) for p, c in pairs]

        self.cache_dir = self.root / ".cache_points"
        if self.cache_npy:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, src_path: str) -> Path:
        p = Path(src_path)
        split_tag = self.split
        base = f"{split_tag}_{p.parent.name}_{p.stem}_{self.npoints}_g{self.gravity_dim}_h{int(self.add_height)}.npy"
        return self.cache_dir / base

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path, label = self.samples[idx]
        if self.cache_npy:
            cp = self._cache_path(path)
            if cp.is_file():
                pts = np.load(cp)
            else:
                pts = _load_points(path, self.npoints)
                # 归一化到零均值/单位球
                pts = pts - pts.mean(axis=0, keepdims=True)
                scale = np.max(np.linalg.norm(pts, axis=1))
                if scale > 0:
                    pts = pts / scale
                np.save(cp, pts)
        else:
            pts = _load_points(path, self.npoints)
            pts = pts - pts.mean(axis=0, keepdims=True)
            scale = np.max(np.linalg.norm(pts, axis=1))
            if scale > 0:
                pts = pts / scale
        if not np.isfinite(pts).all() or pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Corrupted point data: {path}, shape={pts.shape}")
        return {
            "points": torch.from_numpy(pts).float(),  # (N,3)
            "label": int(label),
            "path": path,
        }
