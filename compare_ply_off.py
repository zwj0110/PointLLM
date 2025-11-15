import os
import open3d as o3d

def count_off_vertices(off_path):
    """
    读取 OFF 文件头，返回顶点数 v。
    """
    with open(off_path, 'r') as f:
        first = f.readline().strip().split()
        if first[0] == 'OFF' and len(first) == 4:
            return int(first[1])
        elif first[0] == 'OFF':
            counts = f.readline().strip().split()
            return int(counts[0])
        else:
            raise ValueError(f"Invalid OFF header: {off_path}")

def count_ply_vertices(ply_path):
    """
    用 Open3D 读取 PLY 点云，返回点数。
    """
    pc = o3d.io.read_point_cloud(ply_path)
    return len(pc.points)

# ———— 只对这几对文件做对比 ————
file_pairs = [
    (
      "/Users/zhengwenjie/Downloads/ModelNet40/bed/test/bed_0516.off",
      "/Users/zhengwenjie/Downloads/ModelNet40_ply/bed/test/bed_0516.ply"
    ),
    (
      "/Users/zhengwenjie/Downloads/ModelNet40/airplane/test/airplane_0627.off",
      "/Users/zhengwenjie/Downloads/ModelNet40_ply/airplane/test/airplane_0627.ply"
    ),
    # … 如需更多，继续往里加
]

# 打印表头
print(f"{'file':<40}{'OFF pts':>10}{'PLY pts':>10}{'Δ':>6}")
print("-"*66)

# 逐对统计并打印
for off_path, ply_path in file_pairs:
    off_count = count_off_vertices(off_path)
    ply_count = count_ply_vertices(ply_path)
    delta     = ply_count - off_count
    name      = os.path.basename(off_path)
    print(f"{name:<40}{off_count:10d}{ply_count:10d}{delta:6d}")
