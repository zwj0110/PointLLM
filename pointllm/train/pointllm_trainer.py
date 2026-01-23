import os
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    递归地解包模型，处理分布式训练 (DDP) 中的 module 包装。
    确保保存的权重 Key 名不带 'module.' 前缀。
    """
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


class PointLLMTrainer(Trainer):
    """
    专为 PointLLM 8 维极限压缩任务设计的 Trainer。
    重写 _save：轻量保存，仅保留可训练模块 + 必要配置。
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        拦截原生保存逻辑。不再调用 super()._save() 以免生成 14GB 的大文件。
        """
        # 1) 确定保存目录
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 2) 取模型与 state_dict
        model_to_save = unwrap_model(self.model)

        # ✅ 用 requires_grad 做一层“真实可训练参数”过滤（更稳）
        trainable_param_names = {n for n, p in model_to_save.named_parameters() if p.requires_grad}

        if state_dict is None:
            state_dict = model_to_save.state_dict()

        weight_to_save = {}

        # --- [关键匹配列表] ---
        # 1) pre_proj_adapter: 8 -> 256 的识别路径适配器
        # 2) point_proj: 256 -> hidden 的 LLM 投影层
        # 3) adapter1: 8 -> 768 的蒸馏对齐层
        # 4) point_backbone: GraspNet 学生模型骨干
        keys_to_match = [
            "pre_proj_adapter",
            "point_proj",
            "adapter1",
            "point_backbone",
        ]

        for k, v in state_dict.items():
            # ✅ 双保险：永远不保存 teacher（避免 checkpoint 膨胀/混乱）
            if "teacher_model" in k:
                continue

            # ✅ 必须命中关键词
            if not any(key_match in k for key_match in keys_to_match):
                continue

            # ✅ 只保存可训练参数（避免误存冻结大模块）
            # 注意：state_dict 里也包含 buffer（如 running_mean），这里按名字做过滤最稳
            if k not in trainable_param_names:
                # 但有些可训练模块的 buffer 你也许想保留（通常没有）
                # 如果以后需要，可以在这里放行特定 buffer key
                continue

            weight_to_save[k] = v.detach().cpu()

        if len(weight_to_save) == 0:
            logger.warning(
                "⚠️ Trainer 未匹配到任何可训练参数用于保存。"
                "请检查：requires_grad 是否正确设置、以及 key 名是否包含预期关键词。"
            )

        # 3) 执行保存
        # 仍保留你的命名规则：checkpoint -> adapter_model.bin；最终输出 -> point_proj.bin
        save_name = "adapter_model.bin" if "checkpoint" in output_dir else "point_proj.bin"
        torch.save(weight_to_save, os.path.join(output_dir, save_name))

        # 4) 保存必要元数据
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        model_to_save.config.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(f"✨ [轻量化保存成功] 已保存核心可训练权重至: {os.path.join(output_dir, save_name)}")
        logger.info(f"📦 已跳过全量 Llama 权重与 teacher 权重（如存在）。")

    # 注意：不要在此处定义自定义 compute_loss。
    # 我们希望 Trainer 直接使用 PointLLMLlamaForCausalLM.forward 返回的 loss。
