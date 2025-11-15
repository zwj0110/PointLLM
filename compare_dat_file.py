import pickle, numpy as np, random
from collections import Counter

def summarize(path, sample=200):
    with open(path, "rb") as f:
        pts_list, lbl_list = pickle.load(f)
    n = len(pts_list)
    cnt = Counter(int(x.item()) for x in lbl_list)
    idxs = random.sample(range(n), min(sample, n))
    dup_ratios=[]; norm_means=[]; norm_stds=[]
    for i in idxs:
        A = pts_list[i]
        P = A[:, :3]
        N = A[:, 3:6]
        uniq = np.unique(np.round(P, 6), axis=0)
        dup_ratios.append(1 - len(uniq)/len(P))
        nm = np.linalg.norm(N, axis=1)
        norm_means.append(float(np.mean(nm)))
        norm_stds.append(float(np.std(nm)))
    return {
        "path": path,
        "num_samples": n,
        "dup_ratio_mean": float(np.mean(dup_ratios)),
        "normal_mag_mean": float(np.mean(norm_means)),
        "normal_mag_std_mean": float(np.mean(norm_stds)),
        "label_min": min(cnt), "label_max": max(cnt)
    }

# 路径改成你的两个文件
author_path = "/Users/zhengwenjie/Downloads/modelnet40_test_8192pts_fps.dat"
yours_path = "/Users/zhengwenjie/projects/PointLLM/data/ModelNet40_ply/modelnet40_test_8192pts_fps.dat"

print(summarize(author_path))
print(summarize(yours_path))
