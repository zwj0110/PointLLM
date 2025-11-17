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
from pointllm.eval.timing_utils import attach_timers, DeviceTimer
from pointllm.eval.model_stats import (
    params_by_module,
    pretty_print_params,
    flops_by_module,
    pretty_print_flops,
    group_sum,
)
from pointllm.eval.evaluator import start_evaluation

# 清掉旧 handler
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# dataset
try:
    from pointllm.data import ModelNet  # 原始 .dat
except Exception:
    ModelNet = None

from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]

# ★★★ 这里填你训练好的 adapter ckpt 路径 ★★★
#ADAPTER_CKPT = "./output_pointbert_r01/student_adapter_final.pth"
ADAPTER_CKPT = None


def init_model(args):
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f"[INFO] Model name: {os.path.basename(model_name)}")

    # 设备选择：M1 上优先 mps，其次 cuda，否则 cpu
    device = 'mps' if torch.backends.mps.is_available() else \
             'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 先取 config，写入自定义字段
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg.point_backbone = "PointBERT"
    cfg.point_backbone_ckpt = None
    cfg.mm_use_point_start_end = False
    cfg.fix_pointnet = True

    # ★ 关键：只有 cuda 用 fp16，mps / cpu 用 fp32
    if device == "cuda":
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        config=cfg,
        low_cpu_mem_usage=False,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    ).to(device)

    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    return model, tokenizer, conv



def load_dataset(config_path, split, subset_nums, use_color):
    print(f"Loading {args.split} split of ModelNet datasets.")
    if args.use_dir or ModelNet is None:
        dataset = ModelNet40DirCompat(
            root=args.modelnet_root,
            split=args.split,
            npoints=args.npoints,
            cache_npy=True,
            subset_nums=args.subset_nums,
        )
    else:
        dataset = ModelNet(
            config_path=config_path,
            split=args.split,
            subset_nums=args.subset_nums,
            use_color=args.use_color,
        )
    print("Done!")
    return dataset


def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, (
        "Since we using the index of ModelNet as Object ID when evaluation "
        "so shuffle shoudl be False and should always set random seed."
    )
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)


def generate_outputs(
    model,
    tokenizer,
    input_ids,
    point_clouds,
    stopping_criteria,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    max_length=2048,
    top_p=0.95,
):
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )
    outputs = [o.strip() for o in outputs]
    return outputs


def start_generation(
    model,
    tokenizer,
    conv,
    dataloader,
    prompt_index,
    output_dir,
    output_file,
    timers=None,
):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config["point_token_len"]
    default_point_patch_token = point_backbone_config["default_point_patch_token"]
    mm_use_point_start_end = point_backbone_config.get("mm_use_point_start_end", False)

    if mm_use_point_start_end:
        default_point_start_token = point_backbone_config["default_point_start_token"]
        default_point_end_token = point_backbone_config["default_point_end_token"]
        qs = (
            default_point_start_token
            + default_point_patch_token * point_token_len
            + default_point_end_token
            + "\n"
            + qs
        )
    else:
        qs = default_point_patch_token * point_token_len + "\n" + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    # 统一用模型所在设备
    device = next(model.parameters()).device

    # Tokenizer 计时
    if timers is None:
        timers = {}
    tok_timer = timers.get("tokenizer", DeviceTimer("tokenizer"))
    with tok_timer:
        inputs = tokenizer([prompt])
    timers["tokenizer"] = tok_timer

    input_ids_ = torch.as_tensor(inputs.input_ids).to(device)  # [1, L]

    # FLOPs - 用第一个 batch
    core = model.get_model().eval()
    it = iter(dataloader)
    first_batch = next(it)
    pc1 = first_batch["point_clouds"][:1].to(device).to(model.dtype)
    ids1 = input_ids_[:1]

    try:
        by_mod, flops_total = flops_by_module(core, ids1, pc1)
        pretty_print_flops(by_mod, flops_total, topk=99999)

        prefixes = {
            "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
            "Projector": ["point_proj", "mm_projector", "projector"],
            "LLM": ["language_model", "model", "transformer", "lm"],
        }
        grouped = group_sum(by_mod, prefixes)
        print(
            "Grouped FLOPs (GFLOPs, single forward):",
            {k: v / 1e9 for k, v in grouped.items()},
        )
    except Exception as e:
        print("[WARN] FLOPs analysis failed:", e)

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    # 用剩余 batch 正式推理
    for batch in tqdm(it):
        point_clouds = batch["point_clouds"].to(device).to(model.dtype)
        labels = batch["labels"]
        label_names = batch["label_names"]
        indice = batch["indice"]

        bsz = point_clouds.shape[0]
        input_ids = input_ids_.repeat(bsz, 1)

        outputs = generate_outputs(
            model,
            tokenizer,
            input_ids,
            point_clouds,
            stopping_criteria,
        )

        for index, output, label, label_name in zip(
            indice, outputs, labels, label_names
        ):
            responses.append(
                {
                    "object_id": index.item(),
                    "ground_truth": label.item(),
                    "model_output": output,
                    "label_name": label_name,
                }
            )

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    # 打印计时
    if timers is not None:
        def _fmt(t):
            s = t.stats()
            return f"{s['mean_ms']:.3f} ± {s['std_ms']:.3f} ms (n={s['n']})"

        print("\n===== Inference Component Timing =====")
        if "tokenizer" in timers:
            print("Tokenizer        :", _fmt(timers["tokenizer"]))
        for k, v in timers.items():
            if k == "tokenizer":
                continue
            print(f"{k:16s}: {_fmt(v)}")
        print("======================================\n")

    return results


def main(args):
    # output
    args.output_dir = os.path.join(args.model_name, "evaluation")
    args.output_file = f"ModelNet_classification_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    if not os.path.exists(args.output_file_path):
        # 1) 数据集
        dataset = load_dataset(
            config_path=None,
            split=args.split,
            subset_nums=args.subset_nums,
            use_color=args.use_color,
        )
        dataloader = get_dataloader(
            dataset, args.batch_size, args.shuffle, args.num_workers
        )

        # 2) 模型
        model, tokenizer, conv = init_model(args)
        # core = model.get_model()
        #
        # # ★★★ 在这里插上你训练好的 PointBERT adapter ★★★
        # if hasattr(core, "point_backbone"):
        #     pt = core.point_backbone
        #     print("[INFO] point_backbone type:", type(pt))
        #
        #     if hasattr(pt, "init_adapters"):
        #         pt.init_adapters(
        #             start_layer=pt.depth // 2,  # 和训练脚本一致：后半层
        #             hidden_dim=256,
        #             dropout=0.1,
        #             scale=1.0,
        #         )
        #         print("[INFO] init_adapters() called on point_backbone.")
        #     else:
        #         print("[WARN] point_backbone has no method 'init_adapters'.")
        #
        #     if hasattr(pt, "load_adapter_checkpoint") and ADAPTER_CKPT is not None:
        #         pt.load_adapter_checkpoint(
        #             ADAPTER_CKPT,
        #             only_adapter=True,
        #         )
        #         print(f"[INFO] Adapter checkpoint loaded from {ADAPTER_CKPT}")
        #     else:
        #         print("[WARN] cannot load adapter checkpoint.")
        # else:
        #     print("[WARN] core has no attribute 'point_backbone', cannot attach adapter.")
        core = model.get_model()

        # 不使用任何 adapter，直接用原始 point_backbone
        if hasattr(core, "point_backbone"):
            print("[INFO] Using point_backbone without adapters.")
        else:
            print("[WARN] core has no attribute 'point_backbone'.")

        # 3) 打印参数统计
        rows, total_params = params_by_module(core)
        prefixes = {
            "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
            "Projector": ["point_proj", "mm_projector", "projector"],
            "LLM": ["language_model", "model", "transformer", "lm"],
        }
        grouped_params = group_sum({name: n for name, _, n in rows}, prefixes)
        print(
            "Grouped Params (M):",
            {k: v / 1e6 for k, v in grouped_params.items()},
        )
        timers, _hooks = attach_timers(model, tokenizer)
        print("[Timing] timers attached (tokenizer / point_encoder / projector).")

        # 4) 推理
        print(f"[INFO] Start generating results for {args.output_file}.")
        results = start_generation(
            model,
            tokenizer,
            conv,
            dataloader,
            args.prompt_index,
            args.output_dir,
            args.output_file,
            timers=timers,
        )

        # 5) 释放显存
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[INFO] {args.output_file_path} already exists, directly loading...")
        with open(args.output_file_path, "r") as fp:
            results = json.load(fp)

    # 6) GPT 评估（可选）
    evaluated_output_file = args.output_file.replace(
        ".json", f"_evaluated_{args.gpt_type}.json"
    )
    if args.start_eval:
        start_evaluation(
            results,
            output_dir=args.output_dir,
            output_file=evaluated_output_file,
            eval_type="modelnet-close-set-classification",
            model_type=args.gpt_type,
            parallel=True,
            num_workers=20,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="RunsenXu_graspnet_r04_adapter/PointLLM_7B_v1.2",
    )

    # dataset
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--subset_nums",
        type=int,
        default=-1,
        help="only use 'subset_nums' of samples, mainly for debug",
    )

    # evaluation
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument(
        "--gpt_type",
        type=str,
        default="gpt-3.5-turbo-0613",
        choices=[
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4-0613",
            "gpt-4-1106-preview",
        ],
        help="Type of the model used to evaluate.",
    )

    parser.add_argument(
        "--use_dir",
        action="store_true",
        help="强制使用目录直读（而不是 .dat）",
    )
    parser.add_argument(
        "--modelnet_root",
        type=str,
        default=None,
        help="ModelNet40 根目录（包含 'ModelNet40' 的上级目录，或直接就是 ModelNet40 目录）",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=8192,
        help="目录直读时的采样点数",
    )

    parser.add_argument(
        "--use_color",
        action="store_true",
        default=True,
        help="是否追加伪颜色（全 0 三通道），使 C 从 3 变 6（或 4->8）",
    )
    parser.add_argument(
        "--use_height",
        action="store_true",
        default=False,
        help="是否追加 height 通道（y - y_min），使 C 从 3 变 4",
    )
    parser.add_argument(
        "--gravity_dim",
        type=int,
        default=1,
        help="height 的重力轴维度（0:x, 1:y, 2:z），默认 1",
    )

    args = parser.parse_args()
    main(args)
