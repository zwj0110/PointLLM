import os
import pickle
import numpy as np
import trimesh
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pointllm.data.utils import farthest_point_sample
from pointllm.utils import cfg_from_yaml_file

def _process_one(idx, cid, filepath, npoints):
    """
    单文件处理：读取 .ply -> (N,6) [xyz+normals或0] -> 采样到 npoints
    返回 (idx, pts(npoints,6), label)
    """
    mesh = trimesh.load(filepath, process=False)

    verts = np.asarray(mesh.vertices, dtype=np.float32)  # (V,3)
    if verts.size == 0:
        # 极端情况：空 mesh
        verts = np.zeros((1, 3), dtype=np.float32)

    # 法线可能不存在或数量不一致，统一兜底为 0
    norms = getattr(mesh, "vertex_normals", None)
    if norms is not None and len(norms) == len(verts):
        norms = np.asarray(norms, dtype=np.float32)
    else:
        norms = np.zeros_like(verts, dtype=np.float32)

    pc = np.hstack([verts, norms])  # (V,6)

    # FPS 或随机补采样
    if pc.shape[0] >= npoints:
        sampled = farthest_point_sample(pc, npoints)
    else:
        idxs = np.random.choice(pc.shape[0], npoints, replace=True)
        sampled = pc[idxs, :]

    return idx, sampled.astype(np.float32), np.int64(cid)

def build_modelnet_dat(config_path: str, split: str, num_workers: int = 8):
    """
    并行遍历类别下的 PLY 文件，做 FPS 采样，保存为 .dat（(pts_list, lbl_list)）。
    """
    # 1) 配置
    config       = cfg_from_yaml_file(config_path)
    root         = config["DATA_PATH"]
    npoints      = config.npoints
    num_category = config.NUM_CATEGORY

    # 2) 类别
    config_dir = os.path.dirname(config_path)
    catfile    = os.path.join(config_dir, "modelnet40_shape_names_modified.txt")
    if not os.path.isfile(catfile):
        raise FileNotFoundError(f"找不到类别列表: {catfile}")
    with open(catfile, 'r') as f:
        categories = [line.strip().replace(' ', '_') for line in f]

    # 3) 收集所有 .ply
    file_paths = []
    for cid, cat in enumerate(categories):
        folder = os.path.join(root, cat, split)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith('.ply'):
                file_paths.append((cid, os.path.join(folder, fn)))

    total = len(file_paths)
    print(f"✔ Detected {total} PLY files under '{split}' split.")

    # 4) 并行处理
    pts_list = [None] * total
    lbl_list = [None] * total
    errors = 0

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as ex:
        futures = {
            ex.submit(_process_one, i, cid, fp, npoints): i
            for i, (cid, fp) in enumerate(file_paths)
        }
        for fut in tqdm(as_completed(futures), total=total, desc="Processing files", unit="file"):
            i = futures[fut]
            try:
                idx, pts, lbl = fut.result()
                pts_list[idx] = pts
                lbl_list[idx] = lbl
            except Exception as e:
                errors += 1
                # 标记为空，稍后会过滤
                pts_list[i] = None
                lbl_list[i] = None
                print(f"⚠️ 处理失败: {file_paths[i][1]}\n   {e}")

    # 过滤失败的条目
    ok_pairs = [(p, l) for p, l in zip(pts_list, lbl_list) if p is not None]
    if not ok_pairs:
        raise RuntimeError("没有成功的样本可保存。")
    pts_list, lbl_list = zip(*ok_pairs)

    # 5) 保存 .dat
    save_name = f"modelnet{num_category}_{split}_{npoints}pts_fps.dat"
    save_path = os.path.join(root, save_name)
    with open(save_path, 'wb') as f:
        pickle.dump((list(pts_list), list(lbl_list)), f)

    print(f"✔ 已保存 {len(pts_list)} 个样本到 {save_path}（失败 {errors} 个文件）")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Build ModelNet40 .dat file with parallel processing")
    parser.add_argument(
        "--config_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "modelnet_config/ModelNet40.yaml"),
        help="YAML 配置文件路径"
    )
    parser.add_argument(
        "--split", type=str, choices=["train","test"], default="test",
        help="数据集划分：train 或 test"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="并行线程数（建议 4~16）"
    )
    args = parser.parse_args()

    build_modelnet_dat(args.config_path, args.split, args.num_workers)
