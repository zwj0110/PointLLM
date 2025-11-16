import argparse
import torch
from torch.utils.data import DataLoader

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.eval.timing_utils import attach_timers
from pointllm.eval.timing_utils import DeviceTimer
from pointllm.eval.model_stats import (
    params_by_module, pretty_print_params,
    flops_by_module, pretty_print_flops,
    group_sum
)
import logging, sys
# 如果之前有人配置过 logging，先清掉旧 handler（可选）
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,  # ← 关键
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # 输出到 stdout
)

try:
    from pointllm.data import ModelNet  # 原始 .dat 读取
except Exception:
    ModelNet = None

from pointllm.data.modelnet40_dir_compat import ModelNet40DirCompat
from tqdm import tqdm
from pointllm.eval.evaluator import start_evaluation
from transformers import AutoTokenizer
 # ★改成你的绝对路径


import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of "
]

from transformers import AutoConfig  # 或者你自己的 PointLLMConfig

def init_model(args):
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    device = 'mps' if torch.backends.mps.is_available() else \
             'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 关键：先取 config，先把自定义字段写进去
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    cfg.point_backbone = "PointBERT"
    cfg.point_backbone_ckpt = None
    cfg.point_adapter_ckpt = "/Users/zhengwenjie/projects/PointLLM/output_r04/adapter_epoch5.pth"  # ← 改成你的真实路径
    cfg.mm_use_point_start_end = False
    cfg.fix_pointnet = True

    # 再用带好字段的 config 去构建模型，这样 __init__ 里能读到正确路径
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        config=cfg,                        # ★ 关键
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16 if device == 'mps' else torch.float16,
        trust_remote_code=True
    ).to(device)

    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"
    conv = conv_templates[conv_mode].copy()
    return model, tokenizer, conv


def load_dataset(config_path, split, subset_nums, use_color):
    print(f"Loading {args.split} split of ModelNet datasets.")

    dataset = None
    # 如果强制用目录直读，或者 .dat 模式不可用
    if args.use_dir or ModelNet is None:
        # 这里调用目录直读的兼容类
        dataset = ModelNet40DirCompat(
            root=args.modelnet_root,
            split=args.split,
            npoints=args.npoints,
            cache_npy=True,
            subset_nums=args.subset_nums
        )
    else:
        # 走原始的 .dat 加载
        dataset = ModelNet(
            config_path=args.config_path if hasattr(args, "config_path") else None,
            split=args.split,
            subset_nums=args.subset_nums,
            use_color=args.use_color
        )

    print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
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
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs


def start_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file, timers=None):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    mm_use_point_start_end = point_backbone_config.get('mm_use_point_start_end', False)

    if mm_use_point_start_end:
        default_point_start_token = point_backbone_config['default_point_start_token']
        default_point_end_token   = point_backbone_config['default_point_end_token']
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs


    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    # === Tokenizer 计时 ===
    if timers is None:
        timers = {}
    tok_timer = timers.get("tokenizer", DeviceTimer("tokenizer"))
    with tok_timer:
        inputs = tokenizer([prompt])
    timers["tokenizer"] = tok_timer

    input_ids_ = torch.as_tensor(inputs.input_ids).to('mps')  # tensor of 1, L

    # === FLOPs - 单次前向统计（用首个 batch，B=1）===
    from pointllm.eval.model_stats import flops_by_module, pretty_print_flops, group_sum
    core = model.get_model().eval()

    it = iter(dataloader)  # 先取一个 batch 用来统计 FLOPs
    first_batch = next(it)
    pc1 = first_batch["point_clouds"][:1].to('mps').to(model.dtype)  # B=1
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
        print("Grouped FLOPs (GFLOPs, single forward):",
              {k: v / 1e9 for k, v in grouped.items()})
    except Exception as e:
        print("[WARN] FLOPs analysis failed:", e)

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    # === 用剩下的 batch 正式推理 ===
    for batch in tqdm(it):
        point_clouds = batch["point_clouds"].to('mps').to(model.dtype)
        labels = batch["labels"]
        label_names = batch["label_names"]
        indice = batch["indice"]

        batchsize = point_clouds.shape[0]
        input_ids = input_ids_.repeat(batchsize, 1)

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria)

        # 保存结果
        for index, output, label, label_name in zip(indice, outputs, labels, label_names):
            responses.append({
                "object_id": index.item(),
                "ground_truth": label.item(),
                "model_output": output,
                "label_name": label_name
            })

    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    # === 打印计时汇总 ===
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
    # * ouptut
    args.output_dir = os.path.join(args.model_name, "evaluation")

    # * output file 
    args.output_file = f"ModelNet_classification_prompt{args.prompt_index}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need to generate results first
        dataset = load_dataset(config_path=None, split=args.split, subset_nums=args.subset_nums, use_color=args.use_color) # * defalut config
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
    
        model, tokenizer, conv = init_model(args)
        core = model.get_model()  # PointLLM 的底层模型（包含 Encoder、Projector、LLM）
        rows, total_params = params_by_module(core)
        # pretty_print_params(rows, total_params, topk=999999)

        # 如果想按模块大类聚合（可选）：
        prefixes = {
            "Point Encoder": ["point_backbone", "point_encoder", "backbone"],
            "Projector": ["point_proj", "mm_projector", "projector"],
            "LLM": ["language_model", "model", "transformer", "lm"],
        }
        grouped_params = group_sum({name: n for name, _, n in rows}, prefixes)
        print("Grouped Params (M):", {k: v / 1e6 for k, v in grouped_params.items()})
        timers, _hooks = attach_timers(model, tokenizer)
        print("[Timing] timers attached (tokenizer / point_encoder / projector).")

        # * ouptut
        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, args.prompt_index, args.output_dir, args.output_file, timers = timers)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    # * evaluation file
    evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    # * start evaluation
    if args.start_eval:
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type="modelnet-close-set-classification", model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu_400/PointLLM_7B_v1.2")

    # * dataset type
    parser.add_argument("--split", type=str, default="test", help="train or test.")

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--subset_nums", type=int, default=-1) # * only use "subset_nums" of samples, mainly for debug 

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-3.5-turbo-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")
    parser.add_argument("--use_dir", action="store_true",
                        help="强制使用目录直读（而不是 .dat）")
    parser.add_argument("--modelnet_root", type=str, default=None,
                        help="ModelNet40 根目录（包含 'ModelNet40' 的上级目录，或直接就是 ModelNet40 目录）")
    parser.add_argument("--npoints", type=int, default=8192,
                        help="目录直读时的采样点数")

    # 和 .dat 对齐的两个开关：
    parser.add_argument("--use_color", action="store_true", default=True,
                        help="是否追加伪颜色（全 0 三通道），使 C 从 3 变 6（或 4->8）")
    parser.add_argument("--use_height", action="store_true", default=False,
                        help="是否追加 height 通道（y - y_min），使 C 从 3 变 4")
    parser.add_argument("--gravity_dim", type=int, default=1,
                        help="height 的重力轴维度（0:x, 1:y, 2:z），默认 1")

    args = parser.parse_args()

    main(args)
