import numpy as np
import open3d as o3d

# 1. 加载 glb（其实也是以 mesh 的形式载入）
mesh = o3d.io.read_triangle_mesh("/Users/zhengwenjie/hf-objaverse-v1/glbs/000-011/329684e1e63d4d16bcf519cc9571c1fb.glb", enable_post_processing=True)

# 2. 获取点坐标（N×3）和颜色（如果有的话，N×3）
points = np.asarray(mesh.vertices)
colors = np.asarray(mesh.vertex_colors)  # 若 glb 中含顶点色

print(f"点数 (XYZ): {points.shape[0]}")
if colors.size:
    print(f"每个点的颜色值范围: {colors.min():.3f} – {colors.max():.3f}")
    print(f"颜色数组形状: {colors.shape}")
else:
    print("该模型不包含顶点颜色。")
