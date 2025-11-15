import os
import open3d as o3d

def off_to_ply_pointcloud(off_path, ply_path, ascii=True):
    # 1) 读取三角网格
    mesh = o3d.io.read_triangle_mesh(off_path)
    # 2) 提取所有顶点到点云
    pc = o3d.geometry.PointCloud()
    pc.points = mesh.vertices
    # 3) 写 PLY 点云（ascii or binary）
    o3d.io.write_point_cloud(
        ply_path,
        pc,
        write_ascii=ascii,
        compressed=False
    )
    print(f"[✔] {off_path} → {ply_path}  ({len(pc.points)} pts)")

# ———— 把这儿换成你要测试的 OFF↔PLY 对 ————
file_pairs = [
    (
      "/Users/zhengwenjie/Downloads/ModelNet40/bed/test/bed_0516.off",
      "/Users/zhengwenjie/Downloads/ModelNet40_ply/bed/test/bed_0516.ply"
    ),
    (
      "/Users/zhengwenjie/Downloads/ModelNet40/airplane/test/airplane_0627.off",
      "/Users/zhengwenjie/Downloads/ModelNet40_ply/airplane/test/airplane_0627.ply"
    ),
    # …更多文件对
]

for off_fp, ply_fp in file_pairs:
    os.makedirs(os.path.dirname(ply_fp), exist_ok=True)
    off_to_ply_pointcloud(off_fp, ply_fp, ascii=True)
