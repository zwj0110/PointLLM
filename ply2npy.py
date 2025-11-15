import os
import glob
import open3d as o3d
import numpy as np

def batch_ply2npy(input_dir: str):
    """
    将 input_dir 下所有 .ply 文件转成同名 .npy 文件，
    npy 中保存 (N,6) data——前 3 维是 xyz，后 3 维是 RGB（0–1）。
    """
    ply_paths = glob.glob(os.path.join(input_dir, "*.ply"))
    print(f"Found {len(ply_paths)} PLY files in {input_dir}")

    for ply_path in ply_paths:
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)  # (N,3)

        if pcd.has_colors():
            cols = np.asarray(pcd.colors, dtype=np.float32)  # (N,3) float 0–1
        else:
            # —— 新增：补齐三维全 0 的“假颜色” —— #
            cols = np.zeros((pts.shape[0], 3), dtype=np.float32)
            # —— 结束 —— #

        # 统一拼成 (N,6)
        data = np.hstack([pts, cols])

        base = os.path.splitext(os.path.basename(ply_path))[0]
        npy_path = os.path.join(input_dir, f"{base}.npy")
        np.save(npy_path, data)
        print(f"→ Saved {data.shape} to {npy_path}")

if __name__ == "__main__":
    batch_ply2npy("compressed_ply_dir")
