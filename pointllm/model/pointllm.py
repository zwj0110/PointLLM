from typing import List, Optional, Tuple, Union
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from contextlib import nullcontext
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .utils import *
from pointllm.utils import *
from .adapters import ResidualMLPAdapter

logger = logging.getLogger(__name__)


class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"
    distill_alpha = 1.0  # MSE 权重
    lambda_rec = 1.0  # D 权重
    lambda_rate = 0.01  # R 权重


class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)
        # ... [此处保持你原有的 Backbone 初始化逻辑不变] ...

        # Adapter 3: Pre-projector
        self.pre_proj_adapter = ResidualMLPAdapter(
            dim=self.point_backbone_config["backbone_output_dim"],
            hidden_dim=getattr(config, "pre_proj_adapter_hidden", 256)
        )
        self.point_proj = nn.Linear(self.point_backbone_config["backbone_output_dim"], config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, point_clouds=None, **kwargs):
        inputs_embeds = self.embed_tokens(input_ids)
        compression_metrics = {}

        if self.point_backbone is not None and point_clouds is not None:
            # 1. 教师特征 f (无损)
            with torch.no_grad():
                f_teacher = self.point_backbone.get_original_features(point_clouds)

            # 2. 学生特征 hat_f (压缩重构)
            hat_f_seq, R_loss, D_loss = self.point_backbone(point_clouds, return_loss=True)

            # 3. Pre-proj Adapter
            hat_f_seq = self.pre_proj_adapter(hat_f_seq)

            # 记录用于 Loss 的指标 (去掉 CLS token 对齐维度)
            compression_metrics = {
                'f_teacher': f_teacher,
                'f_hat': hat_f_seq[:, 1:, :],
                'R_loss': R_loss,
                'D_loss': D_loss
            }

            # 4. Projector
            point_features = self.point_proj(hat_f_seq)

            # 5. 拼接逻辑 (简化示意，请使用你原有的长循环替换)
            # [此处插入你代码中 cur_new_input_embeds 的拼接循环逻辑]
            # inputs_embeds = self._process_point_tokens(input_ids, inputs_embeds, point_features)

        outputs = super().forward(input_ids=None, inputs_embeds=inputs_embeds, **kwargs)
        if kwargs.get("return_dict", True):
            outputs['compression_metrics'] = compression_metrics
        return outputs


import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast


# 1. 定义一个扩展的输出类，支持额外的 Loss 字段
@dataclass
class PointLLMOutput(CausalLMOutputWithPast):
    r_loss: Optional[torch.FloatTensor] = None
    d_loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            point_clouds: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, PointLLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. 调用底层的 PointLLMLlamaModel
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            point_clouds=point_clouds,
            return_dict=True,  # 内部强制 True 方便取数据
            **kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        # 初始化辅助损失记录
        r_loss_val, d_loss_val, mse_loss_val = None, None, None

        if labels is not None:
            # A. 标准 LLM 文本损失
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            # B. GRASP 相关损失集成
            # 注意：此处从 outputs 中提取。确保你的 Model 类在 forward 中确实存了这块数据
            metrics = getattr(outputs, 'compression_metrics', None)

            if metrics is not None:
                # 1. 蒸馏损失 (MSE)
                f_hat = metrics.get('f_hat')
                f_teacher = metrics.get('f_teacher')

                if f_hat is not None and f_teacher is not None:
                    mse_loss_val = F.mse_loss(f_hat, f_teacher)
                else:
                    mse_loss_val = torch.tensor(0.0).to(logits.device)

                # 2. 获取 R 和 D 损失
                r_loss_val = metrics.get('R_loss', torch.tensor(0.0).to(logits.device))
                d_loss_val = metrics.get('D_loss', torch.tensor(0.0).to(logits.device))

                # 3. 读取超参数权重
                alpha = getattr(self.config, "distill_alpha", 1.0)
                l_rec = getattr(self.config, "lambda_rec", 1.0)
                l_rate = getattr(self.config, "lambda_rate", 0.01)

                # 4. 加权合并
                loss = loss + (alpha * mse_loss_val) + (l_rec * d_loss_val) + (l_rate * r_loss_val)

        # 5. 返回自定义的 PointLLMOutput
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return PointLLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            r_loss=r_loss_val,
            d_loss=d_loss_val,
            mse_loss=mse_loss_val
        )