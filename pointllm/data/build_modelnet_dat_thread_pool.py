import os
import pickle
import numpy as np
import trimesh
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pointllm.data.utils import farthest_point_sample
from pointllm.utils import cfg_from_yaml_file
from zipfile import ZipFile, ZIP_DEFLATED
import random

def process_one(args):
    cid, cat, filepath, npoints = args
    # 加载点云和法线
    mesh = trimesh.load(filepath, process=False)
    verts = mesh.vertices.astype(np.float32)
    # 有些文件可能没有法线，保险起见处理一下
    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(verts):
        norms = mesh.vertex_normals.astype(np.float32)
    else:
        # 若无法线，填充零
        norms = np.zeros_like(verts, dtype=np.float32)
    pc = np.hstack([verts, norms])  # (V,6)

    # FPS 或随机采样
    if pc.shape[0] >= npoints:
        sampled = farthest_point_sample(pc, npoints)
    else:
        idxs = np.random.choice(pc.shape[0], npoints, replace=True)
        sampled = pc[idxs, :]

    return sampled.astype(np.float32), np.int64(cid)

def collect_file_paths(root, categories, split, per_class_limit, shuffle, seed):
    """
    返回限制后的 (cid, cat, filepath) 列表，以及用于打包 zip 的相对路径列表。
    """
    if shuffle and seed is not None:
        random.seed(seed)

    file_paths = []
    selected_relpaths = []  # 用于 zip：相对 root 的路径

    for cid, cat in enumerate(categories):
        folder = os.path.join(root, cat, split)
        if not os.path.isdir(folder):
            continue
        # 收集该类下所有 .ply
        fns = [fn for fn in os.listdir(folder) if fn.endswith(".ply")]
        if shuffle:
            random.shuffle(fns)
        else:
            fns.sort()

        # 只取前 per_class_limit 个（不足就全取）
        fns = fns[:per_class_limit]

        for fn in fns:
            abs_fp = os.path.join(folder, fn)
            rel_fp = os.path.relpath(abs_fp, root)  # zip 内保持数据集相对结构
            file_paths.append((cid, cat, abs_fp))
            selected_relpaths.append(rel_fp)

    return file_paths, selected_relpaths

def build_modelnet_dat(config_path: str,
                       split: str,
                       num_workers: int = 8,
                       per_class_limit: int = 10,
                       shuffle: bool = False,
                       seed: int = 42,
                       make_zip: bool = True):
    # 1. 加载配置
    config = cfg_from_yaml_file(config_path)
    root = config["DATA_PATH"]
    npoints = config.npoints
    num_category = config.NUM_CATEGORY

    # 2. 加载类别名称
    config_dir = os.path.dirname(config_path)
    catfile = os.path.join(config_dir, "modelnet40_shape_names_modified.txt")
    if not os.path.isfile(catfile):
        raise FileNotFoundError(f"找不到类别列表: {catfile}")
    with open(catfile, 'r') as f:
        categories = [line.strip().replace(' ', '_') for line in f]

    # 3. 收集每类限制后的文件
    file_paths, selected_relpaths = collect_file_paths(
        root, categories, split, per_class_limit, shuffle, seed
    )

    print(f"✔ Detected {len(file_paths)} PLY files under '{split}' split "
          f"(limit {per_class_limit} per class, shuffle={shuffle}).")

    # 4. 并行处理
    pts_list = []
    lbl_list = []
    errors = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, (cid, cat, fp, npoints)): fp for cid, cat, fp in file_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            fp = futures[fut]
            try:
                pts, lbl = fut.result()
                pts_list.append(pts)
                lbl_list.append(lbl)
            except Exception as e:
                errors += 1
                print(f"⚠️ 处理失败: {fp}\n   {e}")

    # 5. 序列化并保存 .dat
    save_name = f"modelnet{num_category}_{split}_{npoints}pts_fps_{per_class_limit}perclass.dat"
    save_path = os.path.join(root, save_name)
    with open(save_path, 'wb') as f:
        pickle.dump((pts_list, lbl_list), f)
    print(f"✔ 已保存 {len(pts_list)} 个样本到 {save_path}（失败 {errors} 个文件）")

    # 6. 生成 zip（包含原始选中的 .ply 文件）
    if make_zip and selected_relpaths:
        zip_name = f"modelnet_originals_{split}_{per_class_limit}perclass.zip"
        zip_path = os.path.join(root, zip_name)
        with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zf:
            for rel in tqdm(selected_relpaths, desc="Zipping originals", unit="file"):
                abs_path = os.path.join(root, rel)
                if os.path.isfile(abs_path):
                    # arcname 保持相对 root 的目录结构
                    zf.write(abs_path, arcname=rel)
        print(f"✔ 已打包原始文件到 {zip_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Build ModelNet40 .dat (subset per class) and zip originals")
    parser.add_argument(
        "--config_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "modelnet_config/ModelNet40.yaml"),
        help="YAML 配置文件路径"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "test"], default="test",
        help="数据集划分：train 或 test"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="线程数"
    )
    parser.add_argument(
        "--per_class_limit", type=int, default=10,
        help="每个类别最多选取的文件数（不足则全取）"
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="是否随机抽样（默认按文件名排序取前 N）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机抽样的随机种子（--shuffle 时生效）"
    )
    parser.add_argument(
        "--no_zip", action="store_true",
        help="不生成包含原始文件的 zip"
    )
    args = parser.parse_args()

    build_modelnet_dat(
        args.config_path,
        args.split,
        args.num_workers,
        args.per_class_limit,
        shuffle=args.shuffle,
        seed=args.seed,
        make_zip=(not args.no_zip)
    )
