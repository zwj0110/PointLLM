#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pip install numpy matplotlib open3d

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
import re

# ------- helpers -------
def load_xyz(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    return np.asarray(pcd.points, dtype=np.float32)

def project(points: np.ndarray, plane: str):
    plane = plane.lower()
    if plane == "xy": return points[:,0], points[:,1]
    if plane == "yz": return points[:,1], points[:,2]
    if plane in ("zx","xz"): return points[:,2], points[:,0]
    raise ValueError("plane must be one of: 'xy','yz','zx'/'xz'")

def union_limits(pairs, margin_ratio=0.03):
    xs = np.concatenate([p[0] for p in pairs])
    ys = np.concatenate([p[1] for p in pairs])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    dx, dy = xmax - xmin, ymax - ymin
    if dx <= 0: dx = 1.0
    if dy <= 0: dy = 1.0
    mx, my = dx*margin_ratio, dy*margin_ratio
    return xmin-mx, xmax+mx, ymin-my, ymax+my

def annotate_below(ax, text, fontsize=12, pad=-0.12):
    ax.annotate(text, xy=(0.5, pad), xycoords="axes fraction",
                ha="center", va="top", fontsize=fontsize)

def auto_labels(orig_label, recon_paths):
    labels = [orig_label]
    for i, p in enumerate(recon_paths, 1):
        m = re.search(r"r0?(\d+)", p, re.I)
        labels.append(f"r{int(m.group(1)):02d}" if m else f"r{i:02d}")
    return labels

# ------- 单视角 → 2×3 网格（不显示CD、不统一点数、已去掉 pts） -------
def make_single_view_grid(
    orig_path: str,
    recon_paths: list,
    plane="xy",                 # "xy"/"yz"/"zx"
    labels=None,                # ["uncompressed","r01","r02","r03","r04","r05"]
    bpps=None,                  # 可选：每列 bpp，没有就传 None 或不传
    out="panel_xy_grid.png",
    ncols=3,                    # 每行 3 张 → 6 张就是 2 行
    col_w=2.0, row_h=2.0, dpi=500,
    point_size=3.2, alpha=0.98,
    bg="white",
    mono_color="#8a8a8a",      # ✅ 统一灰色（可改成你喜欢的灰）
):
    assert len(recon_paths) >= 1, "Need at least one reconstructed file."
    items = [orig_path] + list(recon_paths)
    M = len(items)

    # 标签 & bpp
    if labels is None or len(labels) != M:
        labels = auto_labels("uncompressed", recon_paths)
    if bpps is not None and len(bpps) != M:
        raise ValueError("bpps length must match number of columns (orig + recon).")

    # 读数据 + 投影
    clouds = [load_xyz(p) for p in items]
    xy_pairs = [project(clouds[k], plane) for k in range(M)]
    lims = union_limits(xy_pairs)

    # 网格
    nrows = int(np.ceil(M / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(col_w * ncols, row_h * nrows),
        dpi=dpi, sharex=True, sharey=True, layout="constrained"
    )
    axes = np.array(axes).reshape(-1)
    fig.patch.set_facecolor(bg)

    # 美化
    for ax in axes:
        ax.set_facecolor(bg)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top","right","left","bottom"):
            ax.spines[sp].set_visible(False)

    # 绘制（✅ 不分原始/重建，全部同一灰色）
    for i in range(M):
        x, y = xy_pairs[i]
        ax = axes[i]
        ax.scatter(x, y, s=point_size, c=mono_color, alpha=alpha, linewidths=0)

        # 只显示：label + （可选）bpp   —— 无 CD、无 pts
        parts = [labels[i]]
        if bpps is not None and bpps[i] is not None:
            parts.append(f"{bpps[i]:.2f} bpp")
        annotate_below(ax, " · ".join(parts), fontsize=10, pad=-0.12)

    # 多余格子隐藏
    for j in range(M, len(axes)):
        axes[j].axis("off")

    # 标题（视角）
    axes[0].set_title(f"View: {plane.upper()}", fontsize=10, pad=2.0)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 举例：固定一批文件
    orig = "data/test_datasets_sample/laptop/laptop_0150_xyz_ascii.ply"
    recons = [
        "data/test_datasets_sample/laptop/grasp_laptop_0150_r01.ply",
        "data/test_datasets_sample/laptop/grasp_laptop_0150_r02.ply",
        "data/test_datasets_sample/laptop/grasp_laptop_0150_r03.ply",
        "data/test_datasets_sample/laptop/grasp_laptop_0150_r04.ply",
        "data/test_datasets_sample/laptop/grasp_laptop_0150_r05.ply"
    ]
    labels = ["uncompressed", "r01", "r02", "r03", "r04", "r05"]
    bpps = [None] * 6  # 如果暂时没有 bpp，就全 None

    # 每个方向各导出一张 2×3 图
    make_single_view_grid(
        orig_path=orig, recon_paths=recons, plane="xy",
        labels=labels, bpps=bpps, out="laptop_xy_grid.png",
        ncols=3, col_w=2.0, row_h=2.0, dpi=500,
        point_size=3.2, alpha=0.98, bg="white"
    )
    make_single_view_grid(
        orig_path=orig, recon_paths=recons, plane="yz",
        labels=labels, bpps=bpps, out="laptop_yz_grid.png",
        ncols=3, col_w=2.0, row_h=2.0, dpi=500,
        point_size=3.2, alpha=0.98, bg="white"
    )
    make_single_view_grid(
        orig_path=orig, recon_paths=recons, plane="zx",
        labels=labels, bpps=bpps, out="laptop_zx_grid.png",
        ncols=3, col_w=2.0, row_h=2.0, dpi=500,
        point_size=3.2, alpha=0.98, bg="white"
    )
