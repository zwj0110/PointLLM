import numpy as np
import open3d as o3d

# data/converted_xyz_structured/.cache_points/test_test_guitar_0156_8192_g1_h0.npy
# data/bench_surface_dense_r01_new/.cache_points/test_test_guitar_0156_8192_g1_h0.npy
points = np.load("data/bench_surface_dense_r05_new/.cache_points/test_test_guitar_0158_8192_g1_h0.npy")


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.paint_uniform_color([0.3, 0.3, 0.3])
o3d.visualization.draw_geometries([pcd])
