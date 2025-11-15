#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import random
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import trimesh
from transformers import AutoTokenizer

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.eval.evaluator import start_evaluation

# macOS 更稳：spawn
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]

# ---------- utils: numpy FPS（简洁版） ----------
def farthest_point_sample_np(xyz: np.ndarray, npoints: int) -> np.ndarray:
    """
    xyz: (N, 3) float32
    return indices: (npoints,)
    """
    N = xyz.shape[0]
    if N <= npoints:
        idx = np.arange(N)
        if N < npoints:
            extra = np.random.choice(idx, npoints - N, replace=True)
            idx = np.concatenate([idx, extra], axis=0)
        return idx.astype(np.int64)

    # 初始化：离中心最远的点
    center = xyz.mean(axis=0, keepdims=True)
    dist = np.linalg.norm(xyz - center, axis=1)
    farthest = int(np.argmax(dist))

    selected = np.empty((npoints,), dtype=np.int64)
    min_dist = np.full((N,), np.inf, dtype=np.float32)
    for i in range(npoints):
        selected[i] = farthest
        d = np.linalg.norm(xyz - xyz[farthest], axis=1)
        min_dist = np.minimum(min_dist, d)
        farthest = int(np.argmax(min_dist))
    return selected

# ---------- 只扫描并读取现有的 ModelNet40 目录；在内存里复刻 .dat 预处理 ----------
class ModelNetRawPreproc(Dataset):
    """
    期望目录结构（保持不变）：
      data_root/                    # 这里就是 ModelNet40 根目录
        └── {class}/
            ├── train/*.ply
            └── test/*.ply
    仅扫描并读取；在 __getitem__ 内做：
      - 固定采样（默认 8192，FPS）
      - 中心化 + 单位球归一化（默认开启）
      - 构造 (N,6) = xyz + normals（无法线 → 0）
    """
    def __init__(self, data_root, split='test', subset_nums=-1,
                 npoints=8192, use_color=False, normalize=True, seed=None):
        self.root = Path(data_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data_root 不存在或不是目录: {self.root}")

        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if not classes:
            raise RuntimeError(f"{self.root} 下面没有类别子目录")
        self.class_to_id = {c: i for i, c in enumerate(classes)}
        self.split = split
        self.use_color = bool(use_color)   # 默认 False：避免训练时没用颜色而引入分布偏移
        self.normalize = bool(normalize)
        self.npoints = int(npoints) if npoints and int(npoints) > 0 else 8192

        samples = []
        for cls in classes:
            sd = self.root / cls / split
            if not sd.is_dir():
                continue
            fns = sorted([fn for fn in os.listdir(sd) if fn.lower().endswith(".ply")])
            for fn in fns:
                samples.append((str(sd / fn), self.class_to_id[cls], cls))

        if seed is not None:
            random.Random(seed).shuffle(samples)

        if subset_nums and subset_nums > 0:
            samples = samples[:subset_nums]

        if not samples:
            raise RuntimeError(f"{split=} 下未找到样本。请确认路径与 split。")
        self.samples = samples

    def __len__(self): return len(self.samples)

    @staticmethod
    def _normalize_unit_sphere(xyz: np.ndarray) -> np.ndarray:
        c = xyz.mean(axis=0, keepdims=True)
        xyz = xyz - c
        scale = np.linalg.norm(xyz, axis=1).max()
        if scale < 1e-6: scale = 1e-6
        return xyz / scale

    def __getitem__(self, idx):
        path, cid, cls = self.samples[idx]

        # 只读取，不改造网格；process=False 更稳
        mesh = trimesh.load(path, process=False)

        xyz = np.asarray(mesh.vertices, dtype=np.float32)
        if xyz.size == 0:
            xyz = np.zeros((1, 3), dtype=np.float32)

        # 归一化（复刻 .dat 常见流程）
        if self.normalize:
            xyz = self._normalize_unit_sphere(xyz)

        # 采样到固定 N（FPS）
        sel = farthest_point_sample_np(xyz, self.npoints)
        xyz = xyz[sel, :]

        # 法线（按采样后的索引取对应法线），否则 0
        feat3 = None
        vn = getattr(mesh, "vertex_normals", None)
        if vn is not None and len(vn) == len(np.asarray(mesh.vertices)):
            vn = np.asarray(vn, dtype=np.float32)
            feat3 = vn[sel, :]
        elif self.use_color:
            vc = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
            if vc is not None:
                c = np.asarray(vc, dtype=np.float32)
                if len(c) == len(np.asarray(mesh.vertices)) and c.shape[1] >= 3:
                    feat3 = c[sel, :3]
                    if feat3.max() > 1.0:  # 0-255 → 0-1
                        feat3 = feat3 / 255.0

        if feat3 is None:
            feat3 = np.zeros_like(xyz, dtype=np.float32)

        pts6 = np.hstack([xyz, feat3]).astype(np.float32)  # (N,6)
        pc = torch.from_numpy(pts6)  # (N,6)
        return {
            "point_clouds": pc,
            "labels": torch.tensor(cid),
            "label_names": cls,
            "indice": torch.tensor(idx)
        }

def init_model(args):
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=torch.float32
    ).to('mps')
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    return model, tokenizer, conv

def load_dataset(split, subset_nums, use_color, data_root, normalize, npoints):
    print(f"[FolderMode] Loading {split} FROM FOLDER (replicate .dat preprocessing): {data_root}")
    ds = ModelNetRawPreproc(
        data_root=data_root,
        split=split,
        subset_nums=subset_nums,
        use_color=use_color,      # 默认 False，更贴近训练分布；需要才打开
        normalize=normalize,      # 默认 True
        npoints=npoints
    )
    print("[FolderMode] Dataset:", ds.__class__.__name__, "| num_samples:", len(ds))
    return ds

def get_dataloader(dataset, batch_size):
    # 现在每个 sample 都是固定 N=8192，不需要自定义 collate；单进程最稳
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        persistent_workers=False
    )

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria,
                     do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model.eval()
    with torch.inference_mode():
        try:
            # 清洗 NaN/Inf，防止 generate 阶段崩
            point_clouds = torch.nan_to_num(point_clouds, nan=0.0, posinf=1e6, neginf=-1e6)
            output_ids = model.generate(
                input_ids,
                point_clouds=point_clouds,   # (B, N, 6) 或 (B, 6, N)（见 start_generation）
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                max_length=max_length,
                top_p=top_p,
                stopping_criteria=[stopping_criteria]
            )
        except RuntimeError as e:
            print(f"[Skip] 该 batch 生成时出错，已跳过。错误信息: {e}")
            return []
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]
    return outputs

def start_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file, channels_first=False):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    cfg = model.get_model().point_backbone_config
    point_token_len = cfg['point_token_len']
    default_point_patch_token = cfg['default_point_patch_token']
    default_point_start_token = cfg['default_point_start_token']
    default_point_end_token = cfg['default_point_end_token']
    mm_use_point_start_end = cfg['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    input_ids_ = torch.as_tensor(inputs.input_ids).to('mps')  # (1, L)

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)
    responses = []

    for i, batch in enumerate(tqdm(dataloader, desc="Gen batches")):
        try:
            print(f"[Debug] batch {i+1}/{len(dataloader)}, object_ids = {batch['indice'].tolist()}")
        except Exception:
            pass

        pc = batch["point_clouds"].to('mps').to(model.dtype)  # (B, N, 6)
        if channels_first:
            pc = pc.permute(0, 2, 1).contiguous()             # (B, 6, N) 如果模型需要

        labels = batch["labels"]
        label_names = batch["label_names"]
        indice = batch["indice"]

        B = pc.shape[0]
        input_ids = input_ids_.repeat(B, 1)

        outputs = generate_outputs(model, tokenizer, input_ids, pc, stopping_criteria)
        if not outputs:
            print(f"[Skip batch] indice={list(indice)}")
            continue

        for index, output, label, label_name in zip(indice, outputs, labels, label_names):
            responses.append({
                "object_id": int(index if not torch.is_tensor(index) else index.item()),
                "ground_truth": int(label if not torch.is_tensor(label) else label.item()),
                "model_output": output,
                "label_name": label_name
            })

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"Saved results to {os.path.join(output_dir, output_file)}")
    return results

def main(args):
    args.output_dir = os.path.join(args.model_name, "evaluation")
    args.output_file = f"ModelNet_classification_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    if not os.path.exists(args.output_file_path):
        dataset = load_dataset(
            split=args.split,
            subset_nums=args.subset_nums,
            use_color=args.use_color,
            data_root=args.data_root,
            normalize=(not args.no_normalize),
            npoints=args.npoints
        )
        dataloader = get_dataloader(dataset, args.batch_size)

        model, tokenizer, conv = init_model(args)
        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader,
                                   args.prompt_index, args.output_dir, args.output_file,
                                   channels_first=args.channels_first)

        del model, tokenizer
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    else:
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    if args.start_eval:
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file,
                         eval_type="modelnet-close-set-classification", model_type=args.gpt_type,
                         parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RunsenXu_M40_I/PointLLM_7B_v1.2")

    # folder dataset
    parser.add_argument("--data_root", type=str, required=True,
                        help="ModelNet40 根目录（不改结构）")
    parser.add_argument("--split", type=str, default="test", choices=["train","test"])
    parser.add_argument("--subset_nums", type=int, default=-1)

    # preprocessing（复刻 .dat 习惯）
    parser.add_argument("--npoints", type=int, default=8192, help="固定采样点数（FPS）")
    parser.add_argument("--no_normalize", action="store_true", default=False,
                        help="禁用中心化+单位球归一（默认开启）")
    parser.add_argument("--use_color", action="store_true", default=False,
                        help="仅在无法线时，用 RGB 作为后 3 维（默认关闭，避免训练/推理分布不一致）")
    parser.add_argument("--channels_first", action="store_true", default=False,
                        help="若模型需要 (B,6,N)，打开此开关")

    # loader & eval
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-3.5-turbo-0613",
                        choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"])

    args = parser.parse_args()
    main(args)
