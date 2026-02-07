# -*- coding: utf-8 -*-
"""
Final Evaluation Script for PointLLM with GraspNet Support (EncDec / Adapter modes)
FIXED VERSION:
1. Forces float32 for point inputs to prevent CUDA Index errors in PointNet++.
2. Loads correct model weights instead of forcing base Vicuna.
"""

import argparse
import os
import json
import logging
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM

# ----------------- Path Setup -----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
pccai_parent_dir = os.path.dirname(current_dir)
if pccai_parent_dir not in sys.path:
    sys.path.append(pccai_parent_dir)

# Try import GraspBridge
try:
    from pointllm.grasp_bridge import build_grasp_bridge
except ImportError:
    build_grasp_bridge = None

# ----------------- Logging Setup -----------------
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ----------------- Imports -----------------
try:
    from pointllm.data import ModelNet
except Exception:
    ModelNet = None

from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]


def _normalize_unit_sphere(point_clouds: torch.Tensor) -> torch.Tensor:
    if torch.isnan(point_clouds).any() or torch.isinf(point_clouds).any():
        return torch.zeros_like(point_clouds)

    xyz = point_clouds[:, :, :3].float()
    centroid = torch.mean(xyz, dim=1, keepdim=True)
    xyz = xyz - centroid
    dist = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=-1)), dim=-1, keepdim=True)[0]
    scale = torch.where(dist < 1e-6, torch.ones_like(dist), dist)
    return xyz / scale.unsqueeze(-1)


def init_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    logger.info(f'[INFO] Loading Model from: {model_path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # =================================================================
    # 1. 尝试加载 Config，如果失败则手动构造 (Bypass config.json)
    # =================================================================
    try:
        # 尝试读取本地或线上的 config.json
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        logger.info("✅ 成功加载本地 config.json")
    except Exception as e:
        logger.warning(f"⚠️ 找不到 config.json (错误: {e})")
        logger.warning("🔧 正在手动构造 PointLLM v1.2 (Large) 的配置...")

        # 借用 Vicuna 的基础配置（LLM部分是一样的）
        try:
            cfg = AutoConfig.from_pretrained("lmsys/vicuna-7b-v1.5", trust_remote_code=True)
        except Exception:
            # 如果连 Vicuna 都连不上，手动创建一个空的 LlamaConfig
            from transformers import LlamaConfig
            cfg = LlamaConfig(vocab_size=32000, hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
                              intermediate_size=11008)

    # =================================================================
    # 2. [CRITICAL FIX] 强行覆盖参数 (Force 1152 dim)
    # 无论上面加载了什么 Config，这里都强制重写 PointBERT 参数
    # 确保它能匹配你的 v1.2 权重
    # =================================================================

    # 构造正确的 backbone 配置 (1152 dim, 512 encoder)
    correct_backbone_config = {
        "NAME": "PointTransformer",
        "point_token_len": 512,
        "mm_use_point_start_end": True,
        "projection_hidden_layer": 0,
        "use_color": args.use_color,
        "default_point_patch_token": "<point_patch>",
        "default_point_start_token": "<point_start>",
        "default_point_end_token": "<point_end>",

        # --- 核心修正区域 ---
        "trans_dim": 1152,  # 【关键】权重是 Large 版
        "encoder_dims": 512,  # 【关键】Encoder 也是 Large 版
        "depth": 12,
        "drop_path_rate": 0.1,
        "cls_dim": 40,
        "num_heads": 6,
        "group_size": 32,
        "num_group": 512,
        "point_dims": 6 if args.use_color else 3,
        "use_max_pool": False
    }

    # 注入 PointLLM 必需参数
    cfg.point_backbone = "PointBERT"
    cfg.point_backbone_ckpt = None  # 推理时不需要加载 backbone 预训练权重，因为都在 model.bin 里了
    cfg.point_token_len = 512
    cfg.point_backbone_config = correct_backbone_config

    # 确保 LLM 层的参数对齐
    cfg.hidden_size = 4096

    logger.info("🔧 Config 手动构造完成，准备加载权重...")

    # =================================================================
    # 3. 加载 Tokenizer
    # =================================================================
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except:
        logger.warning("⚠️ 找不到本地 Tokenizer，使用 Vicuna 通用 Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)

    # =================================================================
    # 4. 加载模型权重
    # =================================================================
    logger.info("Loading PointLLM weights...")
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
        trust_remote_code=True
    ).to(device)

    # =================================================================
    # [CRITICAL FIX] 精度冲突终极修复
    # =================================================================
    # 问题：输入必须是 float32 (为了FPS采样)，但模型是 float16。
    # 解决：
    # 1. 强行把视觉编码器 (PointBERT) 转为 float32，让它能吃 float32 的输入。
    model.get_model().point_backbone.float()
    logger.info("🔧 Fixed: PointBackbone cast to float32 for CUDA compatibility.")

    # 2. 注册一个 Hook，在视觉编码器输出时，自动把 float32 转回 float16。
    #    这样后面的 Projector 和 LLM (它们是 float16) 就能正常接收数据了。
    def cast_output_to_half(module, input, output):
        if isinstance(output, torch.Tensor):
            return output.to(torch.float16)
        return output

    model.get_model().point_backbone.register_forward_hook(cast_output_to_half)
    logger.info("🔧 Fixed: Forward Hook registered to cast RunsenXu_graspnet_enc_dec_r02 back to float16.")
    # =================================================================

    # 禁用多余的 neck
    point_backbone = model.get_model().point_backbone
    if hasattr(point_backbone, "transform_neck3d"):
        point_backbone.transform_neck3d = torch.nn.Identity()

    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_dataset(args):
    logger.info(f"Loading {args.split} split of ModelNet datasets.")
    if args.use_dir or ModelNet is None:
        dataset = ModelNet40DirCompat(
            root=args.modelnet_root, split=args.split, npoints=args.npoints,
            cache_npy=True, subset_nums=args.subset_nums
        )
    else:
        dataset = ModelNet(config_path=None, split=args.split, subset_nums=args.subset_nums, use_color=args.use_color)
    logger.info("Done!")
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria):
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            max_length=2048,
            top_p=0.95,
            stopping_criteria=[stopping_criteria]
        )
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    return [output.strip() for output in outputs]


def save_point_cloud_ply(points, filename):
    """辅助函数：保存点云以便调试"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if points.shape[1] >= 6:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for p in points:
            if len(p) >= 6:
                # 假设颜色在 [0, 1] 之间，转回 [0, 255]
                r, g, b = int(p[3] * 255), int(p[4] * 255), int(p[5] * 255)
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {r} {g} {b}\n")
            else:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def start_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file, args, timers=None,
                     grasp_bridge=None):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    # Prompt 构建
    pbc = model.get_model().point_backbone_config
    point_token_len = pbc.get('point_token_len', 512)
    patch_tok = pbc.get('default_point_patch_token', '<point_patch>')

    if pbc.get('mm_use_point_start_end', False):
        qs = pbc['default_point_start_token'] + patch_tok * point_token_len + pbc['default_point_end_token'] + '\n' + qs
    else:
        qs = patch_tok * point_token_len + '\n' + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    input_ids_ = torch.as_tensor(inputs.input_ids).to(model.device)
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    # 创建调试目录
    debug_dir = os.path.join(output_dir, "debug_ply")
    os.makedirs(debug_dir, exist_ok=True)

    # === 推理循环 ===
    for step, batch in enumerate(tqdm(dataloader, desc=f"Inferencing [{args.grasp_mode}]")):
        pc_raw = batch["point_clouds"].to(model.device).float()  # (B, N, 3)
        pc_input_xyz = pc_raw[:, :, :3]

        # =====================================================================
        # [分流逻辑]
        # =====================================================================
        if args.grasp_mode == 'original':
            pc_processed = pc_input_xyz

        elif args.grasp_mode in ['encdec', 'adapter']:
            if grasp_bridge is None:
                raise RuntimeError(f"Mode is '{args.grasp_mode}' but grasp_bridge is not initialized.")

            with torch.no_grad():
                # GraspNet 输出 XYZ (B, N, 3)
                # 注意：如果 grasp_bridge 输出是 0，或者 noise，这里就会出问题
                xyz_hat, _, _ = grasp_bridge(pc_input_xyz)

            # 归一化 (Input to PointLLM)
            pc_processed = _normalize_unit_sphere(xyz_hat)

        else:
            raise ValueError(f"Unknown grasp_mode: {args.grasp_mode}")

        # =====================================================================
        # [CRITICAL FIX] 颜色修正：不要用全0 (Black)，用全0.5 (Grey)
        # =====================================================================
        pc_processed = pc_processed.to(torch.float32)

        if args.use_color and pc_processed.shape[-1] == 3:
            # 之前是 zeros，也就是 RGB=(0,0,0) 纯黑。
            # PointLLM v1.2 可能把黑色理解为某种特殊材质或者看不清。
            # 改成 0.4 或 0.5 (灰色)，能让几何特征更明显。
            colors = torch.ones_like(pc_processed) * 0.4
            pc_final = torch.cat([pc_processed, colors], dim=-1)
        else:
            pc_final = pc_processed

        # [DEBUG] 保存前 3 个 batch 的点云来看看它是圆是扁
        if step < 3:
            debug_path = os.path.join(debug_dir, f"step_{step}_{args.grasp_mode}.ply")
            save_point_cloud_ply(pc_final[0].cpu().numpy(), debug_path)
            if step == 0:
                logger.info(f"💾 [DEBUG] Saved input point cloud to {debug_path}. Please check it in MeshLab!")

        # 准备数据
        labels = batch.get("labels")
        label_names = batch.get("label_names")
        indice = batch.get("indice")
        bs = pc_final.shape[0]
        input_ids = input_ids_.repeat(bs, 1)

        # 生成
        try:
            outputs = generate_outputs(model, tokenizer, input_ids, pc_final, stopping_criteria)
        except RuntimeError as e:
            if "probability tensor" in str(e):
                logger.error(f"❌ [Step {step}] NaN Logits Error. Skipping batch.")
                outputs = ["Error: NaN"] * bs
            else:
                raise e

        # Sample Printing
        if args.print_samples and (step % max(1, args.sample_every) == 0):
            k = min(args.num_sample_print, bs)
            logger.info("-" * 60)
            logger.info(f"[SAMPLE] step={step} | Mode={args.grasp_mode}")
            for i in range(k):
                gt = label_names[i] if label_names is not None else "N/A"
                logger.info(f"  > GT: {gt:<15} | Pred: {outputs[i]}")
            logger.info("-" * 60)

        # 收集结果
        for i in range(bs):
            responses.append({
                "object_id": int(indice[i].item()) if indice is not None else -1,
                "ground_truth": int(labels[i].item()) if labels is not None else -1,
                "model_output": outputs[i],
                "label_name": label_names[i] if label_names is not None else "N/A"
            })

    # 保存
    results = {"prompt": PROMPT_LISTS[prompt_index], "grasp_mode": args.grasp_mode, "results": responses}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)
    logger.info(f"Saved to {os.path.join(output_dir, output_file)}")
    return results


def main(args):
    # 根据 mode 生成输出目录
    model_basename = os.path.basename(os.path.normpath(args.model_name))
    args.output_dir = os.path.join("RunsenXu_graspnet_enc_dec_r01", model_basename, f"evaluation_{args.grasp_mode}")
    args.output_file = f"ModelNet_prompt{args.prompt_index}.json"

    dataset = load_dataset(args)
    dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)

    model, tokenizer, conv = init_model(args)
    timers = None

    # =========================================================================
    # Load Bridge
    # =========================================================================
    grasp_bridge = None

    if args.grasp_mode == 'original':
        logger.info("🟢 Mode: ORIGINAL. GraspNet will NOT be loaded.")

    elif args.grasp_mode in ['encdec', 'adapter']:
        if not args.grasp_ckpt or not args.grasp_config:
            raise ValueError(f"Mode '{args.grasp_mode}' requires --grasp_ckpt and --grasp_config")

        if build_grasp_bridge is None:
            raise ImportError("Cannot import `build_grasp_bridge`. Check pointllm.grasp_bridge.")

        logger.info(f"🔵 Mode: {args.grasp_mode.upper()}. Initializing GraspNet Bridge...")

        use_enc_adapter = (args.grasp_mode == 'adapter')
        adapter_path = args.adapter_path if use_enc_adapter else None

        if use_enc_adapter and not adapter_path:
            logger.warning("⚠️ Mode is adapter but --adapter_path is not provided!")

        grasp_bridge, matched, total = build_grasp_bridge(
            grasp_config=args.grasp_config,
            grasp_ckpt=args.grasp_ckpt,
            device=model.device,
            voxel_size=args.voxel_size,
            npoints=args.npoints,
            adapter_bin_path=adapter_path,
            use_enc_adapter=use_enc_adapter,
            enc_adapter_ratio=args.enc_adapter_ratio
        )
        grasp_bridge.eval()

    else:
        raise ValueError(f"Invalid grasp_mode: {args.grasp_mode}")

    # Start Eval
    start_generation(model, tokenizer, conv, dataloader, args.prompt_index,
                     args.output_dir, args.output_file, args, timers, grasp_bridge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)

    # Dataset
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_nums", type=int, default=-1)
    parser.add_argument("--use_dir", action="store_true", default=False)
    parser.add_argument("--modelnet_root", type=str, default="data/modelnet40_test_all")
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--use_color", action="store_true", default=True)

    # Eval
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--print_samples", action="store_true", default=True)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--num_sample_print", type=int, default=1)

    # Grasp Args
    parser.add_argument("--grasp_mode", type=str, default="original",
                        choices=["original", "encdec", "adapter"])
    parser.add_argument("--grasp_config", type=str, default=None)
    parser.add_argument("--grasp_ckpt", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--enc_adapter_ratio", type=int, default=8)
    parser.add_argument("--voxel_size", type=float, default=0.01)

    args = parser.parse_args()
    main(args)