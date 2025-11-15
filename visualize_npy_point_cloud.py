import numpy as np
import open3d as o3d

# 1. Load the array
data = np.load("colored_8192.npy")
# data.shape == (N,6) 或者 (N,3)

# 2. Split into xyz and optional color
pts = data[:, :3]   # always exists
cols = data[:, 3:6] if data.shape[1] >= 6 else None  # None if no color

# 3. Build Open3D PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

# 4. Only normalize & assign colors if cols 非空
if cols is not None and cols.size > 0:
    # 如果是 0–255 的 uint8，就把它缩到 0–1
    if cols.dtype != np.float64 and cols.max() > 1:
        cols = cols.astype(np.float64) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(cols)
# else: 不赋 pcd.colors，Open3D 会默认用白色

# 5. Visualize
o3d.visualization.draw_geometries([pcd])
