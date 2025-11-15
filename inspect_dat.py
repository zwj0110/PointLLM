#!/usr/bin/env python3
import pickle
import numpy as np
import os

def main():
    fp = "data/modelnet40_data_original/modelnet40_test_8192pts_fps_test1.dat"
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"File not found: {fp}")

    # 1. 反序列化加载 .dat（pickle 保存）
    data = pickle.load(open(fp, 'rb'))
    print(f"Top‐level object type: {type(data)}, length: {len(data) if hasattr(data,'__len__') else '—'}")

    # 2. data 通常是一个长度 2 的 list，[点云, 标签]
    pcs_list = data[0]  # 取第一个元素：点云集合
    print(f"  Raw pcs_list type: {type(pcs_list)}")

    # 3. 如果它是 list，就转成 ndarray
    if isinstance(pcs_list, list):
        pcs = np.array(pcs_list, dtype=np.float32)
    elif isinstance(pcs_list, np.ndarray):
        pcs = pcs_list
    else:
        raise ValueError(f"Unexpected type for pcs_list: {type(pcs_list)}")

    print(f"Point‐cloud array shape: {pcs.shape}  (samples, 8192, 6)")

    # 4. 取出第一个样本：shape (8192,6)
    sample0 = pcs[0]
    print(f"Single sample shape: {sample0.shape}")

    # 5. 只打印后 3 列（这里是法线向量，不是颜色）
    last3 = sample0[:, 3:6]
    print("First 5 rows of last 3 columns (normals):")
    print(last3[:5])

if __name__ == "__main__":
    main()
