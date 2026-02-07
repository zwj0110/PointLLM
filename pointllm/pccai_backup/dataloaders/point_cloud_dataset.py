# Copyright (c) 2010-2022, InterDigital
# All rights reserved.

# See LICENSE under the root folder.

# A generic point cloud dataset wrapper

from torch.utils.data import DataLoader
from .shapenet_part_loader import ShapeNetPart
from .modelnet_loader import ModelNetSimple, ModelNetOctree
from .lidar_loader import LidarSimple, LidarSpherical, LidarOctree
import torch
import numpy as np


# https://github.com/pytorch/pytorch/issues/5059
# Fix numpy random seed issue with multi worker DataLoader
# Multi worker based on process forking duplicates the same numpy random seed across all workers
# Note that this issue is absent with pytorch random operations
def wif(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def get_point_cloud_dataset(dataset_name):
    """List all the data sets in this function for class retrival."""
    print(f"[INFO] dataset_name = {dataset_name}")
    if dataset_name.lower() == 'shapenet_part':
        dataset_class = ShapeNetPart
    elif dataset_name.lower() == 'modelnet_simple':
        dataset_class = ModelNetSimple
    elif dataset_name.lower() == 'modelnet_octree':
        dataset_class = ModelNetOctree
    elif 'simple' in dataset_name.lower():
        dataset_class = LidarSimple
    elif 'spherical' in dataset_name.lower():
        dataset_class = LidarSpherical
    elif 'octree' in dataset_name.lower():
        dataset_class = LidarOctree
    else:
        dataset_class = None
    return dataset_class


def sparse_collate(list_data):
    """A collate function tailored for generating sparse voxels of MinkowskiEngine."""
    list_data = np.vstack(list_data)
    list_data = torch.from_numpy(list_data)
    return list_data


def point_cloud_dataloader(data_config, syntax=None, ddp=False):
    """A wrapper for point cloud datasets."""
    dataset_cls = get_point_cloud_dataset(data_config[0]['dataset'])
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {data_config[0]['dataset']}")
    point_cloud_dataset = dataset_cls(data_config[0], data_config[1], syntax=syntax)

    # ---------- SAFE: 默认不导出，避免覆盖正式 PLY ----------
    EXPORT_ENABLED = False  # 如需调试导出，改为 True
    if EXPORT_ENABLED and getattr(point_cloud_dataset, "split", None) == "test":
        # 导出到独立目录，绝不写回正式 inputs 目录
        out_dir = "exports/_from_loader_debug"
        print(f"[INFO] debug export enabled → {out_dir}")
        # 注：确保 modelnet_loader 里实现了该方法；若没有，请移除或对齐为你的导出 API
        point_cloud_dataset.export_as_seen_to_ply_grouped(
            out_dir,
            binary=True,
            overwrite=True,
            ply_with_batch_first=False,   # 关闭 batch-first，保持常规 PLY 顶点布局
            per_class_limit=999999
        )
    else:
        print(f"[INFO] export disabled (EXPORT_ENABLED={EXPORT_ENABLED})")

    collate_fn = sparse_collate if data_config[0].get('sparse_collate', False) else None
    dl_conf = data_config[0][data_config[1]]

    if ddp:  # for distributed data parallel
        sampler = torch.utils.data.distributed.DistributedSampler(
            point_cloud_dataset, shuffle=dl_conf['shuffle']
        )
        point_cloud_dataloader = DataLoader(
            point_cloud_dataset,
            batch_size=max(1, int(dl_conf['batch_size'] / torch.cuda.device_count())),
            num_workers=max(0, int(dl_conf['num_workers'] / torch.cuda.device_count())),
            persistent_workers=True if dl_conf['num_workers'] > 0 else False,
            worker_init_fn=wif,
            sampler=sampler,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn
        )
    else:
        point_cloud_dataloader = DataLoader(
            point_cloud_dataset,
            batch_size=dl_conf['batch_size'],
            shuffle=dl_conf['shuffle'],
            num_workers=dl_conf['num_workers'],
            persistent_workers=True if dl_conf['num_workers'] > 0 else False,
            worker_init_fn=wif,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn
        )
    return point_cloud_dataset, point_cloud_dataloader
