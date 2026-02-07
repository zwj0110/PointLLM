import os
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def unwrap_model(model: nn.Module) -> nn.Module:
    """解包分布式包装，确保 Key 名纯净"""
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    return model


class PointLLMTrainerInternal(Trainer):
    """
    专为内部适配器架构设计的 Trainer。
    支持 R + D + Feature(MSE) 联合损失监控与轻量化保存
    """

    def _reset_current_losses(self):
        # 每 step 刷新，避免某一步没有返回某个 loss 却沿用上一步的缓存
        self._current_mse = None
        self._current_r = None
        self._current_d = None

    def _extract_attr_or_key(self, outputs: Any, name: str):
        """兼容 ModelOutput / dict"""
        if outputs is None:
            return None
        if hasattr(outputs, name):
            return getattr(outputs, name)
        if isinstance(outputs, dict):
            return outputs.get(name, None)
        return None

    def compute_loss(self, model, inputs, return_outputs=False):
        self._reset_current_losses()

        outputs = model(**inputs)

        # ---- loss ----
        loss = None
        # tuple / list 兼容（有的 forward return (loss, outputs, ...)）
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            # 常见约定：第一个是 loss
            if torch.is_tensor(outputs[0]):
                loss = outputs[0]
            # 第二个可能是 ModelOutput/dict
            out_obj = outputs[1] if len(outputs) > 1 else None
        else:
            out_obj = outputs

        if loss is None:
            # transformers 的 ModelOutput 既支持 .loss 也支持 ["loss"]
            loss = self._extract_attr_or_key(out_obj, "loss")

        if loss is None:
            raise RuntimeError(
                "[PointLLMTrainerInternal] loss is None. "
                "Check: (1) collator provides `labels`; (2) model forward computes and returns `loss`."
            )

        # ---- sub-losses (optional) ----
        mse_loss = self._extract_attr_or_key(out_obj, "mse_loss")
        r_loss = self._extract_attr_or_key(out_obj, "r_loss")
        d_loss = self._extract_attr_or_key(out_obj, "d_loss")

        # 只要是 tensor 就缓存（float / scalar）
        if torch.is_tensor(mse_loss):
            self._current_mse = mse_loss.detach()
        if torch.is_tensor(r_loss):
            self._current_r = r_loss.detach()
        if torch.is_tensor(d_loss):
            self._current_d = d_loss.detach()

        # debug: 仅打印一次 outputs keys
        if int(os.environ.get("POINTLLM_DEBUG", "0")) == 1 and not getattr(self, "_printed_outputs_keys", False):
            try:
                if isinstance(out_obj, dict):
                    logger.info(f"[DEBUG] model outputs keys: {list(out_obj.keys())}")
                else:
                    logger.info(f"[DEBUG] model outputs attrs: {sorted([a for a in dir(out_obj) if 'loss' in a])}")
            except Exception:
                pass
            self._printed_outputs_keys = True

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]) -> None:
        # 让日志里始终有 total loss（Trainer 有时传的是 loss，有时是 learning_rate 等）
        if "loss" in logs and "train/loss_total" not in logs:
            logs["train/loss_total"] = float(logs["loss"])

        # 分项 loss（有就写，没有就不写）
        if getattr(self, "_current_mse", None) is not None:
            # ✅ mse_loss 在你这里表示 feature distill loss
            logs["train/feature_loss"] = float(self._current_mse.item())

        if getattr(self, "_current_d", None) is not None:
            logs["train/reconstruction_d_loss"] = float(self._current_d.item())

        if getattr(self, "_current_r", None) is not None:
            logs["train/bitrate_r_loss"] = float(self._current_r.item())

        super().log(logs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        轻量化保存：仅保存 requires_grad=True 的参数（适配器/投影/point_backbone 等）
        - 不依赖 state_dict key 是否匹配 named_parameters 名字，避免保存漏掉。
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = unwrap_model(self.model)

        # ✅ 最稳：直接从 named_parameters() 抽出 requires_grad=True 的 tensor
        weight_to_save = {}
        for name, param in model_to_save.named_parameters():
            if param is None:
                continue
            if param.requires_grad:
                # 排除 teacher（如果你真的把 teacher 挂进了 model，避免误存）
                if "teacher" in name:
                    continue
                weight_to_save[name] = param.detach().cpu()

        save_name = "adapter_model.bin"
        torch.save(weight_to_save, os.path.join(output_dir, save_name))

        # 保存 tokenizer / config / training args
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        model_to_save.config.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(f"✨ [内部适配器保存成功] 已保存 {len(weight_to_save)} 个 requires_grad 参数项至: {save_name}")
