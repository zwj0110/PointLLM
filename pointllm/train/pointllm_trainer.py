import os
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Optional


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class PointLLMTrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['point_proj', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "point_proj")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'point_proj.bin'))

        super(PointLLMTrainer, self)._save(output_dir, state_dict)

    # 在 PointLLMTrainer 类中添加此方法
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写计算损失的方法，以包含 GRASP 的 R_loss 和 D_loss
        """
        outputs = model(**inputs)

        # 1. 基础的 LLM 交叉熵损失
        loss = outputs.get("loss")

        # 2. 提取 GRASP 带来的辅助损失
        # 假设我们在模型 forward 中将这些损失存入了 outputs 字典
        r_loss = outputs.get("r_loss", None)
        d_loss = outputs.get("d_loss", None)

        if r_loss is not None and d_loss is not None:
            # 权重系数建议：R_loss (比特率) 影响特征稀疏度，D_loss (重建) 影响几何精度
            # 建议初值：R_weight=0.01, D_weight=0.5 (根据实验调整)
            loss = loss + 0.01 * r_loss + 0.5 * d_loss

        return (loss, outputs) if return_outputs else loss