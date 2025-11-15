#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFF → point-cloud PLY (vox12-ready) converter

- Loads OFF triangle meshes
- Uniformly samples N points on the surface
- Optional centering (subtract centroid)
- Optional 12-bit voxelization: map to [0, 4095] and round to integers, with duplicate removal
- Writes vertex-only PLY (ASCII by default)

Dependencies:
  pip install numpy trimesh plyfile

Examples:
  # Batch convert a standard ModelNet40 directory:
  python off_to_vox12_ply.py \
    --src /path/to/ModelNet40 \
    --dst /path/to/ModelNet40_ply \
    --npoints 10000 \
    --centralize \
    --vox12

  # Convert a single OFF file:
  python off_to_vox12_ply.py \
    --file /path/to/model.off \
    --out  /path/to/model_vox12.ply \
    --npoints 10000 \
    --centralize \
    --vox12
"""
import os
import sys
import argparse
import pathlib
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

def write_points_ply(points: np.ndarray, out_path: str, ascii: bool = True):
    """Write (N,3) points to a vertex-only PLY."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    # Ensure float32 or int32 dtype for plyfile
    if points.dtype not in (np.float32, np.int32, np.float64, np.int64):
        points = points.astype(np.float32)
    vertex = np.empty(points.shape[0], dtype=[('x','f4'),('y','f4'),('z','f4')])
    vertex['x'] = points[:,0]
    vertex['y'] = points[:,1]
    vertex['z'] = points[:,2]
    PlyData([PlyElement.describe(vertex, 'vertex')], text=ascii).write(out_path)

def sample_surface_points(off_path: str, npoints: int) -> np.ndarray:
    """Uniformly sample N points from a triangle mesh surface."""
    mesh = trimesh.load(off_path, force='mesh')  # robust to minor OFF variations
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {off_path}")
    pts, _ = trimesh.sample.sample_surface(mesh, npoints)
    return pts.astype(np.float32)

def centralize_points(pc: np.ndarray) -> np.ndarray:
    """Subtract centroid to center the cloud."""
    centroid = pc.mean(axis=0, keepdims=True)
    return (pc - centroid).astype(np.float32)

def map_to_minmax_isotropic(pc: np.ndarray, coord_min: int, coord_max: int, voxelize: bool) -> np.ndarray:
    """Isotropically scale to [coord_min, coord_max] box; optional integer voxelization with dedup."""
    pmin = pc.min()
    pmax = pc.max()
    if pmax <= pmin:
        # Degenerate cloud, return zeros
        return np.zeros_like(pc, dtype=np.int32 if voxelize else np.float32)
    scale = (coord_max - coord_min) / (pmax - pmin)
    pc2 = (pc - pmin) * scale + coord_min
    if voxelize:
        pc2 = np.unique(np.round(pc2).astype(np.int32), axis=0)
    return pc2

def convert_single(off_path: str, out_path: str, npoints: int, centralize: bool, vox12: bool, ascii_ply: bool):
    pc = sample_surface_points(off_path, npoints)
    if centralize:
        pc = centralize_points(pc)
    if vox12:
        pc = map_to_minmax_isotropic(pc, 0, 4095, voxelize=True)  # 12-bit integer grid
    write_points_ply(pc, out_path, ascii=ascii_ply)

def convert_dir(src_root: str, dst_root: str, npoints: int, centralize: bool, vox12: bool, ascii_ply: bool):
    """Convert a ModelNet-style directory: <src>/<class>/(train|test)/*.off"""
    classes = sorted([d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))])
    total = 0
    for cls in classes:
        for split in ("train", "test"):
            src_dir = os.path.join(src_root, cls, split)
            if not os.path.isdir(src_dir):
                continue
            files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".off")])
            if not files:
                continue
            out_dir = os.path.join(dst_root, cls, split)
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            for idx, fname in enumerate(files):
                off_path = os.path.join(src_dir, fname)
                out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".ply")
                try:
                    convert_single(off_path, out_path, npoints, centralize, vox12, ascii_ply)
                    total += 1
                    if total % 100 == 0:
                        print(f"[{total}] {off_path} -> {out_path}")
                except Exception as e:
                    print(f"[WARN] Fail: {off_path} -> {e}")
    print(f"Done. Converted {total} files. Output: {dst_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="ModelNet40 root directory (contains <class>/train|test/*.off).")
    ap.add_argument("--dst", help="Output root directory for PLYs (mirrors class/split).")
    ap.add_argument("--file", help="Convert a single OFF file.")
    ap.add_argument("--out", help="Output PLY path for single file.")
    ap.add_argument("--npoints", type=int, default=10000, help="Sample points per mesh (default: 10000).")
    ap.add_argument("--centralize", action="store_true", help="Subtract centroid before mapping.")
    ap.add_argument("--vox12", action="store_true", help="Map to [0,4095], round to integers, unique dedup.")
    ap.add_argument("--ascii", dest="ascii_ply", action="store_true", help="Write ASCII PLY (default).")
    ap.add_argument("--binary", dest="ascii_ply", action="store_false", help="Write binary PLY.")
    ap.set_defaults(ascii_ply=True)
    args = ap.parse_args()

    # Single-file mode
    if args.file:
        if not args.out:
            raise SystemExit("--out is required when using --file")
        convert_single(args.file, args.out, args.npoints, args.centralize, args.vox12, args.ascii_ply)
        print(f"Converted 1 file -> {args.out}")
        return

    # Directory mode
    if not args.src or not args.dst:
        raise SystemExit("--src and --dst are required for directory mode")
    convert_dir(args.src, args.dst, args.npoints, args.centralize, args.vox12, args.ascii_ply)

if __name__ == "__main__":
    main()
