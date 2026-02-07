import os
import json
import torch
import numpy as np
import hashlib
import copy
import transformers
import trimesh
from torch.utils.data import Dataset
from typing import Dict, List, Optional

# 确保导入 preprocess 函数
from .utils import preprocess_v1, preprocess_multimodal_point_cloud


# --- 1. 辅助函数 ---
def _seed_from_path(path: str) -> int:
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)


def _load_points_from_ply(path: str, npoints: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"磁盘文件缺失: {path}")

    # 强制以 pointcloud 模式加载以提速
    pc = trimesh.load(path, force="pointcloud", process=False)
    pts = np.asarray(pc.vertices if hasattr(pc, "vertices") else pc.points)

    N = pts.shape[0]
    if N == 0:
        raise ValueError(f"文件 {path} 点云为空")

    rng = np.random.default_rng(_seed_from_path(path))
    idx = rng.choice(N, npoints, replace=(N < npoints))
    return pts[idx].astype(np.float32)


# --- 2. 数据集类定义 ---
# object_point_dataset.py

# object_point_dataset.py

class ObjectPointCloudDataset(Dataset):
    def __init__(self, data_path, anno_path, tokenizer, pointnum=8192,
                 conversation_types=None, use_color=True, data_args=None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.pointnum = pointnum
        self.point_indicator = '<point>'
        # ✅ 这里提取出真正的配置字典
        self.point_backbone_config = getattr(data_args, 'point_backbone_config', None)

        with open(anno_path, "r") as f:
            self.list_data_dict = json.load(f)
        self.list_data_dict = [d for d in self.list_data_dict if d.get('conversation_type', 'simple_description') in (
                    conversation_types or ["simple_description"])]

    def __getitem__(self, index):
        item = self.list_data_dict[index]
        path = item.get('point_cloud') or item.get('point_path')
        if not os.path.isabs(path): path = os.path.join(self.data_path, path)

        pts = _load_points_from_ply(path, self.pointnum)
        pts_tensor = torch.from_numpy(self.pc_norm(pts)).float()
        if pts_tensor.shape[-1] == 3:
            pts_tensor = torch.cat([pts_tensor, torch.zeros_like(pts_tensor)], dim=-1)

        raw_conv = copy.deepcopy(item["conversations"])
        for c in raw_conv:
            c['from'] = 'gpt' if str(c['from']).lower() in ['gpt', 'assistant'] else 'human'

        # ✅ 核心修正：传入配置字典 self.point_backbone_config，而不是 data_args 对象
        sources = preprocess_multimodal_point_cloud([raw_conv], self.point_backbone_config, self.point_indicator)
        data_dict = preprocess_v1(sources, self.tokenizer)

        return {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "point_clouds": pts_tensor
        }

    def __len__(self):
        return len(self.list_data_dict)

    def pc_norm(self, pc):
        xyz = pc[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / (m if m > 1e-6 else 1.0)
        return np.concatenate((xyz, pc[:, 3:]), axis=1) if pc.shape[1] > 3 else xyz


# --- 3. 导出函数 ---
def make_object_point_data_module(tokenizer, data_args):
    from .utils import DataCollatorForPointTextDataset
    dataset = ObjectPointCloudDataset(
        data_path=data_args.data_path,
        anno_path=data_args.anno_path,
        pointnum=data_args.pointnum,
        conversation_types=data_args.conversation_types,
        tokenizer=tokenizer,
        use_color=data_args.use_color,
        data_args=data_args
    )
    return {
        "train_dataset": dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForPointTextDataset(tokenizer=tokenizer)
    }