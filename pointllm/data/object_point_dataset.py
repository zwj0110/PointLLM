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
        raise FileNotFoundError(f"文件缺失: {path}")

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
class ObjectPointCloudDataset(Dataset):
    def __init__(self, data_path, anno_path, tokenizer, pointnum=8192,
                 conversation_types=None, use_color=True, data_args=None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.pointnum = pointnum
        self.point_indicator = '<point>'
        self.point_backbone_config = getattr(data_args, 'point_backbone_config', None)

        print(f"📂 正在加载标注文件: {anno_path}")
        with open(anno_path, "r") as f:
            self.list_data_dict = json.load(f)

        self.conversation_types = conversation_types or ("simple_description",)
        self.list_data_dict = [
            d for d in self.list_data_dict
            if d.get('conversation_type', 'simple_description') in self.conversation_types
        ]
        print(f"✅ 数据集加载完成: 共 {len(self.list_data_dict)} 个有效样本。")

    def pc_norm(self, pc):
        xyz = pc[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        if m < 1e-6: m = 1.0
        xyz = xyz / m
        return np.concatenate((xyz, pc[:, 3:]), axis=1) if pc.shape[1] > 3 else xyz

    def __getitem__(self, index):
        curr_idx = index
        attempt = 0
        max_attempts = 100
        last_err = "No error recorded"

        while attempt < max_attempts:
            try:
                item = self.list_data_dict[curr_idx]

                # 1. 路径检查
                path = item['point_cloud']
                if not os.path.isabs(path):
                    path = os.path.join(self.data_path, path)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"磁盘找不到文件: {path}")

                # 2. 点云处理
                pts = _load_points_from_ply(path, self.pointnum)
                pts_tensor = torch.from_numpy(self.pc_norm(pts)).float()
                if pts_tensor.shape[-1] == 3:
                    pts_tensor = torch.cat([pts_tensor, torch.zeros_like(pts_tensor)], dim=-1)

                # 3. 对话清洗
                raw_conv = copy.deepcopy(item["conversations"])
                for c in raw_conv:
                    c['from'] = 'gpt' if str(c['from']).lower() in ['gpt', 'assistant'] else 'human'

                # 4. 预处理 (占位符替换与分词)
                sources = preprocess_multimodal_point_cloud([raw_conv], self.point_backbone_config,
                                                            self.point_indicator)
                data_dict = preprocess_v1(sources, self.tokenizer)

                if data_dict is None or "input_ids" not in data_dict or data_dict["input_ids"].numel() == 0:
                    raise ValueError("分词结果 input_ids 为空")

                # --- 核心改动：构造返回对象 ---
                final_item = {
                    "input_ids": data_dict["input_ids"][0],
                    "labels": data_dict["labels"][0],
                    "point_clouds": pts_tensor
                }

                # ！！！防御性自检：如果是空字典，直接在这里原地自爆，不要传给 Collator ！！！
                if not final_item or "input_ids" not in final_item:
                    raise RuntimeError(f"❌ 逻辑悖论：即将返回空字典！Index: {curr_idx}")

                return final_item

            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)}"
                # print(f"⚠️ 样本 {curr_idx} 加载失败: {last_err}")
                curr_idx = (curr_idx + 1) % len(self.list_data_dict)
                attempt += 1

        raise RuntimeError(f"❌ 连续 {max_attempts} 个样本均加载失败。最后报错: {last_err}")

    def __len__(self):
        return len(self.list_data_dict)


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