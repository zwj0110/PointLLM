import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .adapters import ResidualMLPAdapter

logger = logging.getLogger(__name__)

# HuggingFace / LLaMA 系列里常用的 label mask index
IGNORE_INDEX = -100


@dataclass
class PointLLMOutput(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    mse_loss: Optional[torch.FloatTensor] = None
    r_loss: Optional[torch.FloatTensor] = None
    d_loss: Optional[torch.FloatTensor] = None


class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"

    def __init__(self, distill_alpha=1.0, lambda_rec=1.0, lambda_rate=0.01, **kwargs):
        self.distill_alpha = distill_alpha
        self.lambda_rec = lambda_rec
        self.lambda_rate = lambda_rate
        super(PointLLMConfig, self).__init__(**kwargs)


class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)
        # 动态获取 Token 长度，不再硬编码 512
        self.point_backbone_config = getattr(config, "point_backbone_config", {"point_token_len": 513})
        self.point_backbone = None

        backbone_dim = self.point_backbone_config.get("backbone_output_dim", 8)
        self.intermediate_dim = 256

        # 识别路径映射 (8 -> 256 -> hidden)
        self.pre_proj_adapter = nn.Sequential(
            nn.Linear(backbone_dim, self.intermediate_dim),
            nn.ReLU(),
            ResidualMLPAdapter(self.intermediate_dim, self.intermediate_dim)
        )
        self.point_proj = nn.Linear(self.intermediate_dim, config.hidden_size)

    def forward(self, input_ids=None, attention_mask=None, point_clouds=None, **kwargs):
        # 1) 原始文本 embedding
        inputs_embeds = self.embed_tokens(input_ids)
        compression_metrics = None

        if self.point_backbone is not None and point_clouds is not None:
            # 2) teacher features (no grad)
            with torch.no_grad():
                f_teacher = self.point_backbone.get_original_features(point_clouds)

            # student forward: [B, N, 8] + (R, D)
            hat_f_seq, R_loss, D_loss = self.point_backbone(point_clouds, return_loss=True)

            # 3) 维度对齐：严格遵循配置 point_token_len
            target_n = int(self.point_backbone_config.get("point_token_len", 513))
            if hat_f_seq.dim() == 2:
                # [B,8] -> [B,N,8]
                hat_f_seq = hat_f_seq.unsqueeze(1).repeat(1, target_n, 1)
            elif hat_f_seq.shape[1] != target_n:
                # [B,*,8] -> [B,N,8]
                hat_f_seq = F.interpolate(hat_f_seq.transpose(1, 2), size=target_n).transpose(1, 2)

            # 4) 映射到 LLM hidden 维度
            hat_f_seq_256 = self.pre_proj_adapter(hat_f_seq.float())
            point_features = self.point_proj(hat_f_seq_256).to(inputs_embeds.dtype)  # [B,N,hidden]

            # 5) 更稳健：按 <point_patch> 的 token id 精确替换（如果配置提供了 id）
            patch_id = self.point_backbone_config.get("point_patch_token_id", None)

            if patch_id is None:
                # 兜底：沿用旧假设（前 N 个位置是 point slots）
                if inputs_embeds.size(1) < target_n:
                    raise ValueError(f"inputs_embeds length {inputs_embeds.size(1)} < target_n {target_n}")
                inputs_embeds = torch.cat([point_features, inputs_embeds[:, target_n:, :]], dim=1)
            else:
                mask = (input_ids == int(patch_id))  # [B,L]
                counts = mask.sum(dim=1)
                if torch.any(counts < target_n):
                    raise ValueError(
                        f"Not enough <point_patch> tokens: min={counts.min().item()} < target_n={target_n}. "
                        f"Check your data_module tokenization / point_token_len."
                    )

                inputs_embeds = inputs_embeds.clone()
                # 替换每条样本的前 target_n 个 patch 位置
                for b in range(input_ids.size(0)):
                    idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)[:target_n]  # [target_n]
                    inputs_embeds[b, idx, :] = point_features[b, :target_n, :]

            compression_metrics = {
                "f_teacher": f_teacher,
                "f_hat": hat_f_seq,
                "R_loss": R_loss,
                "D_loss": D_loss,
            }

        outputs = super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        if compression_metrics is not None:
            setattr(outputs, "compression_metrics", compression_metrics)
        return outputs


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(PointLLMLlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, point_clouds=None, **kwargs) -> PointLLMOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, point_clouds=point_clouds, **kwargs)
        logits = self.lm_head(outputs[0])

        loss = None
        ce_loss = torch.tensor(0.0, device=logits.device)
        r_loss_val = torch.tensor(0.0, device=logits.device)
        d_loss_val = torch.tensor(0.0, device=logits.device)
        mse_loss_val = torch.tensor(0.0, device=logits.device)

        if labels is not None:
            # 1) CE（忽略 -100）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            ce_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            ).float()
            loss = ce_loss

            # 2) 蒸馏 + R/D
            metrics = getattr(outputs, "compression_metrics", None)
            if metrics is not None:
                f_hat = metrics.get("f_hat")        # [B,N,8]
                f_teacher = metrics.get("f_teacher")  # [B,1,768] 或 [B,N,768]

                student_backbone = self.model.point_backbone
                if f_hat is not None and f_teacher is not None and hasattr(student_backbone, "adapter1"):
                    # student -> teacher space: [B,N,8] -> [B,N,768]
                    f_hat_projected = student_backbone.adapter1(f_hat.float())

                    # teacher 是全局 [B,1,768]：对齐 student token 的均值
                    if f_teacher.dim() == 3 and f_teacher.shape[1] == 1:
                        f_hat_avg = f_hat_projected.mean(dim=1, keepdim=True)  # [B,1,768]
                        mse_loss_val = F.mse_loss(f_hat_avg, f_teacher.float())
                    else:
                        mse_loss_val = F.mse_loss(f_hat_projected, f_teacher.float())

                    loss = loss + (self.config.distill_alpha * mse_loss_val)

                # 3) R/D（确保是 tensor）
                r_raw = metrics.get("R_loss", None)
                d_raw = metrics.get("D_loss", None)

                if torch.is_tensor(r_raw):
                    r_loss_val = r_raw.to(device=logits.device).mean()
                else:
                    r_loss_val = torch.tensor(0.0, device=logits.device)

                if torch.is_tensor(d_raw):
                    d_loss_val = d_raw.to(device=logits.device).mean()
                else:
                    d_loss_val = torch.tensor(0.0, device=logits.device)

                loss = loss + (self.config.lambda_rec * d_loss_val) + (self.config.lambda_rate * r_loss_val)

            # 更可调试：打印各项加权贡献
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(
                    f"\r[TRAIN] "
                    f"CE:{ce_loss.item():.3f} "
                    f"| a*MSE:{(self.config.distill_alpha*mse_loss_val).item():.3f} "
                    f"| rec*D:{(self.config.lambda_rec*d_loss_val).item():.3f} "
                    f"| rate*R:{(self.config.lambda_rate*r_loss_val).item():.3f} "
                    f"| Total:{loss.item():.3f}",
                    end=""
                )

        return PointLLMOutput(
            loss=loss,
            logits=logits,
            mse_loss=mse_loss_val,
            r_loss=r_loss_val,
            d_loss=d_loss_val,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_model(self):
        return self.model
