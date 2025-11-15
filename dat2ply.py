import numpy as np

def load_modelnet40_dat(dat_path):
    """
    读取一个形如 'modelnet40_test_8192pts_fps_test1.dat' 的二进制文件，
    假设它按 float32 连续写入了 M 个 (8192, 6) 样本（每个样本 8192 个点，每点 6 维：xyz+法线）。
    返回：
      - data_xyz: 形状 (M, 8192, 3) 的 numpy.ndarray
      - data_normals: 形状 (M, 8192, 3) 的 numpy.ndarray
      - M: 实际读取到的样本数
    """
    # 1. 第一遍读出所有 float32
    raw = np.fromfile(dat_path, dtype=np.float32)
    total_floats = raw.size
    print(f"→ 从 '{dat_path}' 一共读到 {total_floats} 个 float32 元素。")

    # 2. 每个样本占用 num_per_sample = 8192 * 6 个 float
    num_per_sample = 8192 * 6
    M = total_floats // num_per_sample  # 实际能整除的样本数
    if M == 0:
        raise ValueError("数据量太少，无法 reshape 为 (8192, 6) 结构，请确认 dat 文件是否正确。")

    # 3. 如果尾部有“多余”没凑够 8192*6 的浮点数，我们直接丢弃
    usable_floats = M * num_per_sample
    if usable_floats != total_floats:
        print(f"  注意：文件共有 {total_floats} 个 float，"
              f"但其中只有前 {usable_floats} 个可被完整分成 {M} 个 (8192×6)，"
              f"剩余 {total_floats - usable_floats} 个 float 会被舍弃。")

    # 4. reshape 成 (M, 8192, 6)
    raw = raw[:usable_floats]
    data_all = raw.reshape((M, 8192, 6))

    # 5. 分离坐标和法线
    data_xyz = data_all[:, :, :3]      # (M, 8192, 3)
    data_normals = data_all[:, :, 3:6] # (M, 8192, 3)
    return data_xyz, data_normals, M

# 例子：测试如何调用
if __name__ == "__main__":
    dat_path = "/data/modelnet40_data_original/modelnet40_test_8192pts_fps_test1.dat"  # 请改成你本地实际路径
    xyz, normals, num_models = load_modelnet40_dat(dat_path)
    print(f"最后确认：一共成功解析了 {num_models} 个 点云样本。")
