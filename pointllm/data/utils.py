from collections import OrderedDict, defaultdict
import transformers
from pointllm import conversation as conversation_lib
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List, Any
import torch
import numpy as np
import os
import random
IGNORE_INDEX = -100


class LRUCache:
    def __init__(self, capacity, max_access_count):
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.capacity = capacity
        self.max_access_count = max_access_count

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)
            del self.access_count[oldest_key]
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        self.access_count[key] = 0


def preprocess_v1(sources, tokenizer):
    # 1. 强制获取 Vicuna 1.1 模板，确保角色名和分隔符准确
    conv = conversation_lib.conv_templates["vicuna_v1_1"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 构造原始对话长文本
    conversations = []
    for i, source in enumerate(sources):
        if source[0]["from"] not in roles:
            source[0]["from"] = "human"
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence["from"], conv.roles[j % 2])
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # 2. Tokenize 转化为张量
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # 3. 逐样本处理 Mask (Label)
    for i, (conversation, target) in enumerate(zip(conversations, targets)):
        # 按照 </s> 分割轮次（处理多轮对话）
        rounds = conversation.split(conv.sep2)
        cur_len = 1  # 跳过开头的 BOS (<s>)
        target[:cur_len] = IGNORE_INDEX

        for j, rou in enumerate(rounds):
            if rou == "": break

            # --- 核心加固：动态匹配分隔符 ---
            # 针对你的 JSON 格式，处理 USER 和 ASSISTANT 之间的空格或换行
            possible_seps = [
                " " + conv.roles[1] + ":",  # " ASSISTANT:"
                "\n" + conv.roles[1] + ":",  # "\nASSISTANT:"
                conv.roles[1] + ":"  # "ASSISTANT:"
            ]

            parts = None
            actual_sep = None
            for s in possible_seps:
                if s in rou:
                    parts = rou.split(s)
                    actual_sep = s
                    break

            # 如果这一轮没搜到分隔符，说明格式异常，屏蔽整轮 Loss
            if parts is None or len(parts) != 2:
                # [DEBUG] 偶尔打印提示
                if random.random() < 0.01:
                    print(f"⚠️ [DEBUG] 分解失败：在内容中找不到角色分隔符。内容: {rou[:30]}...")

                # 尝试计算当前轮次的长度并跳过
                temp_len = len(tokenizer(rou).input_ids) - 1
                cur_len += temp_len + (len(tokenizer(conv.sep2).input_ids) - 1)
                continue

            # 4. 计算长度并屏蔽指令部分 (Human)
            # 这里的逻辑是：指令 = [Human部分] + [ASSISTANT:]
            parts[0] += actual_sep

            round_len = len(tokenizer(rou).input_ids) - 1
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # 将 Human 的提问部分设为 -100 (不计算 Loss)
            mask_end = min(cur_len + instruction_len, target.shape[0])
            target[cur_len: mask_end] = IGNORE_INDEX

            # 累加索引位置，移动到下一轮
            cur_len += round_len
            cur_len += (len(tokenizer(conv.sep2).input_ids) - 1)

        # 5. 屏蔽末尾所有 Padding 字符
        if cur_len < target.shape[0]:
            target[cur_len:] = IGNORE_INDEX

    # --- [DEBUG] 链路透视（仅在第一个样本且随机触发，防止日志过多） ---
    if random.random() < 0.05:
        # 统计有效训练 Token 数量
        valid_label_count = target.ne(IGNORE_INDEX).sum().item()
        print(f"\n--- [DEBUG 数据透视] ---")
        print(f"有效可学习 Token 数: {valid_label_count}")
        if valid_label_count > 0:
            # 解码模型真正“背诵”的内容
            learned_content = tokenizer.decode(target[target != IGNORE_INDEX])
            print(f"模型正在学习的内容: {learned_content[:100]}...")
        else:
            print(f"❌ 严重警告：当前样本有效 Label 为 0，请检查分隔符匹配！")
        print(f"--- [DEBUG 结束] ---\n")

    return dict(input_ids=input_ids, labels=targets)

def preprocess_multimodal_point_cloud(
        sources: Sequence[str],
        point_backbone_config: dict,
        point_indicator: str = "<point>",
) -> Dict:
    # 增加默认值防止 Key 丢失导致的静默失败
    point_token_len = point_backbone_config.get('point_token_len', 513)
    patch_token = point_backbone_config.get('default_point_patch_token', '<point_patch>')

    # 构造替换长字符串
    replace_token = patch_token * point_token_len

    if point_backbone_config.get('mm_use_point_start_end', False):
        start_t = point_backbone_config.get('default_point_start_token', '<point_start>')
        end_t = point_backbone_config.get('default_point_end_token', '<point_end>')
        replace_token = start_t + replace_token + end_t

    replaced_count = 0
    for source in sources:
        for sentence in source:
            if point_indicator in sentence["value"]:
                # [DEBUG] 记录替换前的长度
                # print(f"PRE: {len(sentence['value'])}")
                sentence["value"] = sentence["value"].replace(point_indicator, replace_token)
                replaced_count += 1

    # [DEBUG] 打印确认信息
    if replaced_count > 0:
        print(f"✅ [DEBUG] Multimodal: Replaced '{point_indicator}' in {replaced_count} sentences.")

    return sources



@dataclass
class DataCollatorForPointTextDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 严格过滤：必须是字典、不能为 None、必须包含 input_ids 且不能为空张量
        valid_instances = [
            inst for inst in instances
            if inst is not None and isinstance(inst, dict) and 'input_ids' in inst
        ]

        # 2. 极端情况处理：如果整个 Batch 都坏了
        if len(valid_instances) == 0:
            # 这里的预览会更有用
            print(f"❌ Batch 彻底失败。收到数据类型: {[type(i) for i in instances]}")
            if len(instances) > 0 and isinstance(instances[0], dict):
                print(f"第一个字典的键: {instances[0].keys()}")
            raise ValueError("当前 Batch 所有样本均预处理失败！请检查 Dataset 是否返回了有效字典。")

        instances = valid_instances
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        # 4. 文本序列 Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)  # 确保 IGNORE_INDEX = -100 已在文件开头定义

        # 5. 构造输出 Batch
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'point_clouds' in instances[0]:
            point_clouds = [instance['point_clouds'] for instance in instances]
            if all(x.shape == point_clouds[0].shape for x in point_clouds):
                batch['point_clouds'] = torch.stack(point_clouds)
                # [DEBUG] 极其重要的形状检查
                print(
                    f"🚀 [DEBUG] Final Batch shapes -> Input: {batch['input_ids'].shape}, Points: {batch['point_clouds'].shape}")
            else:
                batch['point_clouds'] = point_clouds
                print(f"⚠️ [DEBUG] Variable Point Clouds detected in batch.")

        return batch


def pc_norm(pc):
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    # 防止除以 0
    if m < 1e-6: m = 1.0
    xyz = xyz / m
    return np.concatenate((xyz, other_feature), axis=1)


def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    filename = f"{object_id}_{pointnum}.npy"
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        # 调试路径用
        print(f"❌ 找不到点云文件: {full_path}")
        return None

    point_cloud = np.load(full_path)
    point_cloud = pc_norm(point_cloud)
    if not use_color:
        point_cloud = point_cloud[:, :3]
    return point_cloud


def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
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
    return point[centroids.astype(np.int32)]