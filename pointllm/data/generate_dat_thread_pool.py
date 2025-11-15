#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简版：不读取配置、不打包原始文件，只生成 .dat
目录要求：
  data_root/
    └── ModelNet40/
        └── {class}/
            ├── train/*.ply
            └── test/*.ply

输出：
  {output}/modelnet{num_classes}_{split}_{npoints}pts_fps_{per_class_limit}perclass.dat
"""

import os
import re
import pickle
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import trimesh
from tqdm import tqdm

# ---------------- FPS(简易版) ----------------
def farthest_point_sample(x: np.ndarray, npoints: int) -> np.ndarray:
    """
    x: (N, C) 这里只需要前3维为坐标
    返回：选中的 npoints 行构成的新数组
    纯 numpy 的简易 FPS，适合 N 不太大的场景
    """
    N = x.shape[0]
    if N <= npoints:
        # 不足则重复采样到 npoints
        idx = np.arange(N)
        if N < npoints:
            extra = np.random.choice(idx, npoints - N, replace=True)
            idx = np.concatenate([idx, extra], axis=0)
        return x[idx, :]

    xyz = x[:, :3]
    cent = np.mean(xyz, axis=0, keepdims=True)  # 初始点尽量离中心远一点
    dist = np.linalg.norm(xyz - cent, axis=1)
    farthest = np.argmax(dist)

    selected = []
    min_dist = np.full((N,), np.inf, dtype=np.float32)

    for _ in range(npoints):
        selected.append(farthest)
        d = np.linalg.norm(xyz - xyz[farthest], axis=1)
        min_dist = np.minimum(min_dist, d)
        farthest = int(np.argmax(min_dist))

    return x[np.array(selected, dtype=np.int64), :]

# ---------------- 数据收集 ----------------
def collect_file_paths(root: Path, split: str, per_class_limit: int, shuffle: bool, seed: int):
    """
    root: 指向包含各 class 子目录的 ModelNet40 目录
    返回：[(cid, class, abs_path), ...]
    """
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"在 {root} 下没有发现类别子目录。")

    if shuffle and seed is not None:
        random.seed(seed)

    file_paths = []
    for cid, cls in enumerate(classes):
        folder = root / cls / split
        if not folder.is_dir():
            continue
        fns = [fn for fn in os.listdir(folder) if fn.lower().endswith(".ply")]
        fns.sort()
        if shuffle:
            random.shuffle(fns)
        fns = fns[:per_class_limit]
        for fn in fns:
            file_paths.append((cid, cls, str(folder / fn)))

    return classes, file_paths

# ---------------- 单文件处理 ----------------
def process_one(args):
    cid, cls, filepath, npoints = args
    mesh = trimesh.load(filepath, process=False)
    verts = mesh.vertices.astype(np.float32)

    # 处理法线（没有则补零）
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None \
       and len(mesh.vertex_normals) == len(verts):
        norms = mesh.vertex_normals.astype(np.float32)
    else:
        norms = np.zeros_like(verts, dtype=np.float32)

    pc = np.hstack([verts, norms])  # (V,6)

    # FPS 或重复采样
    if pc.shape[0] >= npoints:
        sampled = farthest_point_sample(pc, npoints)
    else:
        idxs = np.random.choice(pc.shape[0], npoints, replace=True)
        sampled = pc[idxs, :]

    return sampled.astype(np.float32), np.int64(cid)

# ---------------- 主逻辑 ----------------
def main():
    ap = argparse.ArgumentParser("Build ModelNet40 .dat (no config, only .dat)")
    ap.add_argument("--data_root", type=str, required=True,
                    help="数据集根目录（包含 ModelNet40/xxx 的目录）或直接是 ModelNet40 目录")
    ap.add_argument("--split", type=str, choices=["train", "test"], default="test",
                    help="数据集划分：train 或 test")
    ap.add_argument("--npoints", type=int, default=8192,
                    help="每个样本采样点数")
    ap.add_argument("--per_class_limit", type=int, default=10,
                    help="每个类别最多选取的文件数（不足则全取）")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="并行线程数")
    ap.add_argument("--shuffle", action="store_true",
                    help="是否随机抽样（默认按文件名排序取前 N）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机抽样的随机种子（--shuffle 时生效）")
    ap.add_argument("--output", type=str, default=None,
                    help="输出 .dat 的目录（默认：写到 ModelNet40 目录下）")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    # 兼容传入 data_root/ 或 data_root/ModelNet40
    modelnet_root = data_root / "ModelNet40" if (data_root / "ModelNet40").is_dir() else data_root
    if not modelnet_root.is_dir():
        raise FileNotFoundError(f"未找到 ModelNet40 目录：{modelnet_root}")

    classes, file_paths = collect_file_paths(
        modelnet_root, args.split, args.per_class_limit, args.shuffle, args.seed
    )
    num_classes = len(classes)
    print(f"✔ 类别数：{num_classes} | 采样点：{args.npoints} | split={args.split} | 每类至多 {args.per_class_limit} 个")
    print(f"✔ 待处理文件数：{len(file_paths)}")

    pts_list, lbl_list = [], []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(process_one, (cid, cls, fp, args.npoints)): fp for cid, cls, fp in file_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            fp = futures[fut]
            try:
                pts, lbl = fut.result()
                pts_list.append(pts)
                lbl_list.append(lbl)
            except Exception as e:
                errors += 1
                print(f"⚠️ 失败：{fp}\n   {e}")

    # 输出路径
    out_dir = Path(args.output).expanduser().resolve() if args.output else modelnet_root
    out_dir.mkdir(parents=True, exist_ok=True)

    save_name = f"modelnet{num_classes}_{args.split}_{args.npoints}pts_fps_{args.per_class_limit}perclass.dat"
    save_path = out_dir / save_name

    with open(save_path, "wb") as f:
        pickle.dump((pts_list, lbl_list), f)

    print("\n====== 完成 ======")
    print(f"✔ 成功：{len(pts_list)} | 失败：{errors}")
    print(f"✔ 已保存：{save_path}")

if __name__ == "__main__":
    main()
