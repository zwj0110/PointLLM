#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point-BERT style ModelNet40 preprocessor (with --split-subdir & schema=tuple2)
=============================================================================

- Normalize to unit sphere (center to centroid, scale by max radius)
- Downsample to 8192 points with FPS (pad by random if too few)
- Multiprocessing + progress bar + resumable per-sample cache
- Deterministic ordering & seeds
- Output schemas:
    1) pointbert (default): {'<split>', '<split>_label', '<split>_name'}
    2) tuple2: (points, labels)  <-- matches PointLLM `modelnet.py` expectation

Usage (PointLLM expects tuple2 + D=6)
-------------------------------------
python pointbert_dat_builder_split.py \
  --data-root /PATH/ModelNet40 \
  --class-as-parent \
  --split-subdir test \
  --split-name test \
  --include-normals \
  --npoints 8192 \
  --workers 12 \
  --schema tuple2 \
  --out /PATH/data/ModelNet40/modelnet40_test_8192pts_fps.dat

Notes
-----
* If you omit --include-normals, D=3 → file ~half the size; with normals D=6 → size ≈ 460MB for test.
* tuple2 schema writes pickle((data, labels)).

"""

from __future__ import annotations
import os
import sys
import json
import time
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import trimesh
except Exception:
    print("[Error] trimesh is required: pip install trimesh", file=sys.stderr)
    raise

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------- config ----------------------------- #
class Config:
    def __init__(self,
                 data_root: Path,
                 out_path: Path,
                 class_as_parent: bool = True,
                 split_name: str = "test",
                 split_subdir: str = "",
                 include_normals: bool = False,
                 npoints: int = 8192,
                 workers: int = max(1, os.cpu_count() // 2),
                 seed: int = 42,
                 device: str = "cpu",
                 resume: bool = True,
                 schema: str = "pointbert",  # or 'tuple2'
                 tmp_dir: Optional[Path] = None):
        self.data_root = data_root
        self.out_path = out_path
        self.class_as_parent = class_as_parent
        self.split_name = split_name
        self.split_subdir = split_subdir.strip()
        self.include_normals = include_normals
        self.npoints = npoints
        self.workers = workers
        self.seed = seed
        self.device = device
        self.resume = resume
        self.schema = schema
        self.tmp_dir = tmp_dir


# ----------------------------- geom utils ----------------------------- #

def _ensure_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    norms = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if norms is None or norms.shape[0] != mesh.vertices.shape[0]:
        mesh.rezero()
        norms = np.asarray(mesh.vertex_normals, dtype=np.float32)
    return norms


def normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0, keepdims=True)
    pts_c = pts - c
    r = np.linalg.norm(pts_c, axis=1)
    mr = np.max(r)
    if mr < 1e-8:
        return pts_c
    return pts_c / mr


def _fps_numpy(points: np.ndarray, npoints: int, seed: Optional[int] = None) -> np.ndarray:
    V = points.shape[0]
    if V <= npoints:
        idx = np.arange(V)
        if V < npoints:
            rng = np.random.default_rng(seed)
            extra = rng.choice(V, npoints - V, replace=True)
            idx = np.concatenate([idx, extra])
        return idx
    rng = np.random.default_rng(seed)
    coords = points[:, :3]
    idxs = np.empty(npoints, dtype=np.int64)
    dists = np.full(V, np.inf, dtype=np.float64)
    last = int(rng.integers(V))
    idxs[0] = last
    for i in range(1, npoints):
        diff = coords - coords[last]
        d2 = np.einsum('ij,ij->i', diff, diff)
        np.minimum(dists, d2, out=dists)
        last = int(np.argmax(dists))
        idxs[i] = last
    return idxs


def _fps_torch(points: np.ndarray, npoints: int, device: str = "cuda", seed: Optional[int] = None) -> np.ndarray:
    if not _HAS_TORCH:
        return _fps_numpy(points, npoints, seed)
    V = points.shape[0]
    if V <= npoints:
        idx = np.arange(V)
        if V < npoints:
            g = np.random.default_rng(seed)
            extra = g.choice(V, npoints - V, replace=True)
            idx = np.concatenate([idx, extra])
        return idx
    dev = torch.device(device if device in ("cpu","cuda") else ("cuda" if torch.cuda.is_available() else "cpu"))
    with torch.no_grad():
        coords = torch.from_numpy(points[:, :3]).to(dev)
        V = coords.shape[0]
        idxs = torch.empty(npoints, dtype=torch.long, device=dev)
        dists = torch.full((V,), float('inf'), device=dev)
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(int(seed))
        last = torch.randint(low=0, high=V, size=(1,), generator=g, device=dev).item()
        idxs[0] = last
        for i in range(1, npoints):
            diff = coords - coords[last]
            d2 = torch.sum(diff * diff, dim=1)
            dists = torch.minimum(dists, d2)
            last = torch.argmax(dists).item()
            idxs[i] = last
        return idxs.cpu().numpy()


def downsample(points: np.ndarray, npoints: int, device: str, seed: Optional[int]) -> np.ndarray:
    V = points.shape[0]
    if V == npoints:
        return points
    if V < npoints:
        rng = np.random.default_rng(seed)
        extra = rng.choice(V, npoints - V, replace=True)
        return np.concatenate([points, points[extra]], axis=0)
    idx = _fps_torch(points, npoints, device=device, seed=seed) if (device == 'cuda') else _fps_numpy(points, npoints, seed)
    return points[idx]


def _read_mesh_points(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mesh = trimesh.load(path.as_posix(), process=False)
    if isinstance(mesh, trimesh.Scene):
        comps = [m for m in mesh.dump().values() if isinstance(m, trimesh.Trimesh)]
        if not comps:
            return np.empty((0, 3), dtype=np.float32), None
        mesh = trimesh.util.concatenate(tuple(comps))
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    norms = None
    try:
        norms = _ensure_normals(mesh).astype(np.float32)
    except Exception:
        norms = None
    return verts, norms


def _find_mesh_files(root: Path, exts=(".off", ".ply", ".obj", ".stl", ".glb", ".gltf")) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


# ------------------------------ worker ------------------------------ #

def _process_one(fpath: Path, label_id: int, cfg: Config) -> Tuple[np.ndarray, int, str]:
    verts, norms = _read_mesh_points(fpath)
    if verts.size == 0:
        raise ValueError(f"Empty mesh: {fpath}")
    verts = normalize_unit_sphere(verts)
    if cfg.include_normals and norms is not None and norms.shape[0] == verts.shape[0]:
        pts = np.hstack([verts, norms]).astype(np.float32)  # [V,6]
    else:
        pts = verts.astype(np.float32)                      # [V,3]
    pts = downsample(pts, cfg.npoints, device=cfg.device, seed=cfg.seed)
    return pts, int(label_id), fpath.stem


# ------------------------------- build ------------------------------- #

def build(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # discover classes and tasks
    if cfg.class_as_parent:
        class_dirs = sorted([p for p in cfg.data_root.iterdir() if p.is_dir()])
        classes = [p.name for p in class_dirs]
        classes_sorted = sorted(classes)
        label_map: Dict[str,int] = {c: i for i, c in enumerate(classes_sorted)}
        tasks: List[Tuple[Path,int]] = []
        for d in class_dirs:
            cls = d.name
            if cls not in label_map:
                continue
            lab = label_map[cls]
            base = d
            if cfg.split_subdir:
                cand = d / cfg.split_subdir
                if not cand.exists():
                    continue
                base = cand
            for f in _find_mesh_files(base):
                tasks.append((f, lab))
    else:
        base = cfg.data_root
        if cfg.split_subdir:
            cand = cfg.data_root / cfg.split_subdir
            if cand.exists():
                base = cand
        files = _find_mesh_files(base)
        tasks = [(f, 0) for f in files]
        label_map = {"default": 0}

    total = len(tasks)
    if total == 0:
        raise SystemExit("No files found. Check --data-root and --split-subdir.")

    print(f"[INFO] total files to process: {total} (split_subdir='{cfg.split_subdir or '-'}')")

    # prepare cache dir
    tmp_dir = cfg.tmp_dir or (cfg.out_path.parent / (cfg.out_path.stem + "_tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # resume set
    done = {p.stem for p in tmp_dir.glob('*.npz')} if cfg.resume else set()

    # index csv
    idx_path = cfg.out_path.parent / f"{cfg.out_path.stem}_index.csv"
    index_csv = idx_path.open("w", encoding="utf-8")
    index_csv.write("name,cls,label,relpath")

    errors = 0
    with ProcessPoolExecutor(max_workers=max(1, int(cfg.workers))) as ex:
        futures = {}
        for fpath, lab in tasks:
            if fpath.stem in done:
                continue
            futures[ex.submit(_process_one, fpath, lab, cfg)] = (fpath, lab)

        pbar = tqdm(total=total - len(done), ncols=100, desc="Processing") if tqdm else None
        for fut in as_completed(futures):
            fpath, lab = futures[fut]
            try:
                pts, lab2, name = fut.result()
                np.savez_compressed(tmp_dir / f"{name}.npz", points=pts, label=np.int64(lab2), name=name)
                rel = fpath.relative_to(cfg.data_root)
                index_csv.write(f"{name},{fpath.parent.name if cfg.class_as_parent else 'default'},{lab},{rel.as_posix()}")
            except Exception as e:
                errors += 1
                print(f"[WARN] {fpath}: {e}", file=sys.stderr)
            finally:
                if pbar: pbar.update(1)
        if pbar: pbar.close()
    index_csv.close()

    # pack cached -> chosen schema
    cached = sorted(tmp_dir.glob('*.npz'))
    if not cached:
        raise SystemExit("No cached samples to pack.")

    sample = np.load(cached[0])
    P = cfg.npoints
    D = int(sample['points'].shape[1])

    N = len(cached)
    data = np.empty((N, P, D), dtype=np.float32)
    label = np.empty((N,), dtype=np.int64)
    names: List[str] = []

    it = tqdm(cached, desc="Packing", ncols=100) if tqdm else cached
    for i, npz in enumerate(it):
        d = np.load(npz)
        data[i] = d['points']
        label[i] = d['label'].astype(np.int64)
        names.append(str(d['name']))

    if cfg.schema == 'tuple2':
        # EXACTLY what PointLLM's modelnet.py expects: (points, labels)
        out_obj = (data, label)
    else:
        out_obj = {cfg.split_name: data, f"{cfg.split_name}_label": label, f"{cfg.split_name}_name": names}

    with open(cfg.out_path, 'wb') as f:
        pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        'split': cfg.split_name,
        'split_subdir': cfg.split_subdir,
        'npoints': cfg.npoints,
        'dims': D,
        'include_normals': cfg.include_normals,
        'schema': cfg.schema,
        'class_as_parent': cfg.class_as_parent,
        'seed': cfg.seed,
        'device': cfg.device,
        'date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'errors': int(errors),
        'total_samples': int(N),
    }
    (cfg.out_path.parent / f"{cfg.out_path.stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(f"Saved {N} samples -> {cfg.out_path}")
    if errors:
        print(f"Completed with {errors} warnings.")
    print(f"Index: {idx_path.as_posix()}")


# ----------------------------- CLI ----------------------------- #

def parse_args(argv=None) -> Config:
    import argparse
    ap = argparse.ArgumentParser(description="Point-BERT style ModelNet40 preprocessor (with --split-subdir & schema)")
    ap.add_argument('--data-root', required=True, help='Dataset root (e.g., /path/ModelNet40)')
    ap.add_argument('--out', required=True, help='Output .dat path')
    ap.add_argument('--class-as-parent', action='store_true', help='Treat subfolders as classes')
    ap.add_argument('--split-name', default='test', help="'train' or 'test' (keys in .dat when schema=pointbert)")
    ap.add_argument('--split-subdir', default='', help="Only traverse class/<split-subdir>/* (e.g. 'test' or 'train')")
    ap.add_argument('--include-normals', action='store_true', help='Concatenate vertex normals if available (xyz+normals)')
    ap.add_argument('--npoints', type=int, default=8192, help='Target points per sample')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 2), help='#process workers')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    ap.add_argument('--schema', choices=['pointbert','tuple2'], default='pointbert')
    ap.add_argument('--no-resume', action='store_true', help='Do not resume from cache')
    ap.add_argument('--tmp-dir', default='', help='Optional cache dir for per-sample npz')
    args = ap.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    tmp_dir = Path(args.tmp_dir).expanduser().resolve() if args.tmp_dir else None

    return Config(
        data_root=data_root,
        out_path=out_path,
        class_as_parent=bool(args.class_as_parent),
        split_name=str(args.split_name),
        split_subdir=str(args.split_subdir),
        include_normals=bool(args.include_normals),
        npoints=int(args.npoints),
        workers=int(args.workers),
        seed=int(args.seed),
        device=str(args.device),
        resume=not bool(args.no_resume),
        schema=str(args.schema),
        tmp_dir=tmp_dir,
    )


if __name__ == '__main__':
    cfg = parse_args()
    build(cfg)
