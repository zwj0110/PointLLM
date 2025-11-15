import os
import trimesh
import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# 从 pointllm/data/utils.py 中复制的 farthest_point_sample 函数
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def process_category(args):
    """
    一个工作函数，用于处理一个类别的所有文件。
    返回该类别所有点云和标签的列表。
    """
    category, data_root, split, npoints, use_normals, cat_idx = args

    category_points = []
    category_labels = []
    modified_category_name = category.replace(' ', '_')

    category_dir = os.path.join(data_root, modified_category_name, split)
    if not os.path.exists(category_dir):
        # 目录不存在时，返回空列表
        print(f"Directory not found: {category_dir}. Skipping category: {category}")
        return [], []

    for filename in os.listdir(category_dir):
        if filename.endswith('.off'):
            filepath = os.path.join(category_dir, filename)
            try:
                mesh = trimesh.load(filepath)

                if len(mesh.vertices) == 0:
                    continue

                if use_normals:
                    points = np.hstack((mesh.vertices, mesh.vertex_normals))
                else:
                    points = mesh.vertices

                if points.shape[0] > npoints:
                    points = farthest_point_sample(points, npoints)
                else:
                    indices = np.random.choice(points.shape[0], npoints, replace=True)
                    points = points[indices]

                category_points.append(points.astype(np.float32))
                category_labels.append(np.array(cat_idx, dtype=np.int64))

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

    return category_points, category_labels


def generate_modelnet_dat_multiprocess(data_root, num_category, npoints, split, use_normals=False):
    """
    使用多进程生成 ModelNet 数据集 .dat 文件
    """
    print(f"Generating {split} data for ModelNet{num_category} using multiprocessing...")

    # 请确保 modelnet_config 目录和文件存在于脚本的相对路径中
    catfile = os.path.join(os.path.dirname(__file__), 'modelnet_config', 'modelnet40_shape_names_modified.txt')
    categories = [line.rstrip() for line in open(catfile)]

    list_of_points = []
    list_of_labels = []

    # 准备多进程的参数
    tasks = [(category, data_root, split, npoints, use_normals, i) for i, category in enumerate(categories)]

    # 创建进程池，使用与CPU核心数相同的进程数
    num_processes = cpu_count()
    print(f"Using {num_processes} processes.")
    with Pool(processes=num_processes) as pool:
        # 使用 pool.map() 将任务分配给进程池
        # tqdm 用于显示进度条
        results = list(tqdm(pool.imap(process_category, tasks), total=len(tasks), desc="Processing Categories"))

    # 合并所有进程的结果
    for points, labels in results:
        list_of_points.extend(points)
        list_of_labels.extend(labels)

    # 保存文件
    save_path = os.path.join(data_root, f'modelnet{num_category}_{split}_{npoints}pts_fps.dat')
    with open(save_path, 'wb') as f:
        pickle.dump((list_of_points, list_of_labels), f)

    print(f"Successfully generated {len(list_of_points)} samples and saved to {save_path}")


if __name__ == '__main__':
    # 配置参数
    DATA_ROOT = "/Users/zhengwenjie/projects/PointLLM/data/ModelNet40"  # <--- 替换为你的数据根目录
    NUM_CATEGORY = 40
    NPOINTS = 8192
    USE_NORMALS = True

    # 运行多进程脚本生成训练集和测试集文件
    # generate_modelnet_dat_multiprocess(DATA_ROOT, NUM_CATEGORY, NPOINTS, 'train', USE_NORMALS)
    generate_modelnet_dat_multiprocess(DATA_ROOT, NUM_CATEGORY, NPOINTS, 'test', USE_NORMALS)