import numpy as np
import open3d as o3d

# Load the point data (e.g., shape (8192, 6) or (2048, 3))
data = np.load("data/customize_data/loot_vox10_1200.npy")
print("Data shape:", data.shape)

# Inspect dimensions
if data.ndim == 2:
    n_points, dims = data.shape
    print(f"Number of points: {n_points}")
    print(f"Point dimensions: {dims} ({'3 = xyz only' if dims == 3 else '6 = xyz + RGB'})")

    if dims == 6:
        xyz = data[:, :3]
        rgb = data[:, 3:]
        print(f"RGB range: min={rgb.min():.3f}, max={rgb.max():.3f}")
    else:
        print("No RGB color channels detected.")
else:
    print("Unexpected data dimensions (not a 2D array). Please verify the file.")

# Prepare point cloud for visualization
pts = data[:, :3]
colors = data[:, 3:6] if data.shape[1] == 6 else None

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

if colors is not None:
    # Normalize to [0,1] if values are in 0-255 range
    if colors.dtype == np.uint8 or colors.max() > 1.0:
        colors = colors.astype(float) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

# Display the point cloud
o3d.visualization.draw_geometries([pcd])
