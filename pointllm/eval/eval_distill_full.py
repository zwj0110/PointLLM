import os
import sys
import argparse
import torch
import json
import yaml
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

# --- [路径注入：解决 pccai 导入报错] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
pointllm_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
if pointllm_root not in sys.path:
    sys.path.append(pointllm_root)

# 导入必要的模块
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.pccai.models.architectures.grasp import GeoResCompression
from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat


def _load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)


def init_full_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = args.checkpoint_path  # 指向 checkpoint-3500 文件夹

    # 1. 直接从 checkpoint 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)

    # 2. 加载架构并注入学生网络
    print(f"[INFO] 正在从全量目录初始化架构: {ckpt_path}")
    # 先加载 Config
    model = PointLLMLlamaForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,  # 必须为 False 以支持自定义架构注入
        trust_remote_code=True
    ).to(device)

    # 3. 必须复现训练时的注入逻辑，否则 adapter1 等权重没地方放
    print("[INFO] 正在注入学生网络架构以匹配权重文件...")
    raw_grasp_cfg = _load_yaml(args.grasp_config)
    student = GeoResCompression(raw_grasp_cfg.get("net_config", raw_grasp_cfg), SimpleNamespace(phase="test")).to(
        device)
    student.adapter1 = torch.nn.Sequential(
        torch.nn.Linear(8, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 768)
    ).to(device)
    model.get_model().point_backbone = student

    # 4. 再次加载权重 (这一次会填充进注入的 student 和 adapter1)
    # 注意：因为 save_pretrained 保存了全量状态，直接 load 即可
    print("[INFO] 正在执行全量权重同步...")
    model.from_pretrained(ckpt_path)

    # 5. 注入推理时的 forward 逻辑
    def student_forward_eval(self, point_clouds):
        res = GeoResCompression.forward(self, point_clouds)
        y_hat = res.get('y_hat') if isinstance(res, dict) else res[0]
        if y_hat.dim() == 2: y_hat = y_hat.view(point_clouds.shape[0], -1, y_hat.shape[-1])
        if y_hat.shape[-1] == 1: y_hat = y_hat.repeat(1, 1, 8)

        target_n = 513  # 必须匹配训练日志中的 pt_len
        if y_hat.shape[1] < target_n:
            pad = torch.zeros((y_hat.shape[0], target_n - y_hat.shape[1], 8), device=y_hat.device)
            y_hat = torch.cat([y_hat, pad], dim=1)
        elif y_hat.shape[1] > target_n:
            y_hat = y_hat[:, :target_n, :]

        return self.adapter1(y_hat.float()).half()

    import types
    student.forward = types.MethodType(student_forward_eval, student)

    # 6. 配置同步
    full_p_cfg = {
        "point_token_len": 513,
        "backbone_output_dim": 8,
        "default_point_patch_token": "<point_patch>",
        "mm_use_point_start_end": False  # 根据日志设为 False
    }
    model.get_model().point_backbone_config = full_p_cfg

    return model, tokenizer


def main(args):
    model, tokenizer = init_full_model(args)
    model.eval()

    dataset = ModelNet40DirCompat(root=args.modelnet_root, split="test", npoints=8192)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("[INFO] 开始全量模型推理...")
    for batch in tqdm(dataloader):
        pc = batch["point_clouds"].to(model.device).float()
        conv = conv_templates["vicuna_v1_1"].copy()

        # 匹配训练格式：513 个 patch，无 start/end 标签
        point_placeholder = "<point_patch>" * 513
        qs = point_placeholder + "\nWhat is this?"

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(input_ids, point_clouds=pc, do_sample=True, temperature=0.2, max_length=1024)

        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"Sample - GT: {batch['label_names'][0]} | Pred: {output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint-3500 文件夹路径")
    parser.add_argument("--grasp_config", type=str, required=True)
    parser.add_argument("--modelnet_root", type=str, required=True)
    args = parser.parse_args()
    main(args)