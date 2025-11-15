import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def plot_projection(ply_file, ply_name, ckpoint, plane="xy", sample_size=50000, point_size=2.0):
    """
    将点云投影到二维平面并显示颜色

    参数:
        ply_file: str, 点云文件路径 (.ply)
        plane: str, 投影平面，可选 "xy", "xz", "yz"
        sample_size: int, 随机采样点数量
        point_size: float, 点的大小
    """
    # 读取点云
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # 如果没有颜色，就用默认颜色（黑色）
    if colors.size == 0:
        colors = None

    # 随机采样，避免太卡
    if sample_size < len(points):
        idx = np.random.choice(len(points), size=sample_size, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    # 选择投影平面
    if plane == "xy":
        x, y = points[:, 0], -points[:, 1]
        xlabel, ylabel = "X", "Y"
    elif plane == "xz":
        x, y = points[:, 0], points[:, 2]
        xlabel, ylabel = "X", "Z"
    elif plane == "yz":
        x, y = points[:, 1], points[:, 2]
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("plane must be one of 'xy', 'xz', 'yz'")
    out_file = f"{ply_name}_{ckpoint}_{plane}.png"
    # 绘制
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=point_size, c="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Projection {ckpoint} onto {plane.upper()} plane")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.show()

ply_file = "data/test_datasets_sample/laptop/grasp_laptop_0150_r05.ply"
ckpoint = "R05"
ply_name = "bathtub_0107"

# 示例：可改成自己的 ply 文件路径
plot_projection(ply_file, ply_name=ply_name,ckpoint=ckpoint, plane="xy", point_size=3.0)
plot_projection(ply_file, ply_name=ply_name, ckpoint=ckpoint, plane="xz", point_size=3.0)
plot_projection(ply_file, ply_name=ply_name, ckpoint=ckpoint, plane="yz", point_size=3.0)
