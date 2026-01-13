#    Copyright 2023 Runsen Xu

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .utils import *
from pointllm.utils import *

from contextlib import nullcontext
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

import os

# * add logger
import logging
logger = logging.getLogger(__name__)


class PointLLMConfig(LlamaConfig):
    model_type = "pointllm"


class PointLLMLlamaModel(LlamaModel):
    config_class = PointLLMConfig

    def __init__(self, config: LlamaConfig):
        super(PointLLMLlamaModel, self).__init__(config)

        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        if self.point_backbone_type == "PointBERT":
            from pointllm.model import PointTransformer

            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(
                config,
                "point_backbone_config_name",
                "PointTransformer_8192point_2layer",  # * default for v1.2
            )
            point_bert_config_addr = os.path.join(
                os.path.dirname(__file__),
                "pointbert",
                f"{point_bert_config_name}.yaml",
            )
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)

            # 是否使用颜色通道
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6

            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False)  # * default is false
            print(f"user_max_pool is {use_max_pool}.")

            # 初始化 PointTransformer backbone
            self.point_backbone = PointTransformer(
                point_bert_config.model,
                use_max_pool=use_max_pool,
            )
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            # 记录 backbone / projector 的配置
            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": (
                    point_bert_config.model.trans_dim
                    if not use_max_pool
                    else point_bert_config.model.trans_dim * 2
                ),
                "project_output_dim": self.config.hidden_size,
                # with cls token when not max-pool
                "point_token_len": (
                    point_bert_config.model.num_group + 1
                    if not use_max_pool
                    else 1
                ),
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get(
                    "projection_hidden_layer", 0
                ),
                "use_max_pool": use_max_pool,
            }
            if point_bert_config.model.get("projection_hidden_layer", 0) > 0:
                # a list, e.g. [1024, 2048]
                self.point_backbone_config["projection_hidden_dim"] = (
                    point_bert_config.model.projection_hidden_dim
                )

            logger.info(
                f"Use max pool is {use_max_pool}. "
                f"Number of point token is {self.point_backbone_config['point_token_len']}."
            )

            # ====== 这里只挂 TransformNeck3D 模块，不在 meta 阶段加载权重 ======
            try:
                from pointllm.model.transform_neck3d import TransformNeck3D

                neck_in_dim = point_bert_config.model.trans_dim
                self.point_backbone.transform_neck3d = TransformNeck3D(
                    in_dim=neck_in_dim
                )
                logger.info(
                    f"[Adapter] transform_neck3d attached: "
                    f"{self.point_backbone.transform_neck3d}"
                )
            except Exception as e:
                logger.error(f"[Adapter] Failed to init TransformNeck3D module: {e}")
                self.point_backbone.transform_neck3d = None
        elif self.point_backbone_type == "GRASP":
            # --- build GRASP token backbone ---
            from pointllm.model.grasp_backbone import GraspTokenBackbone, GraspBackboneArgs

            # 你需要在 config 里提供 grasp_model（已加载好权重的 GeoResCompression）
            # 例如在外层 build 模型时：model.model.point_backbone = your_grasp_model
            grasp_model = None
            logger.info("[GRASP] grasp_model will be injected after model initialization (train.py).")
            # 复用 PointBERT 的 num_group（保持 token_len 与 prompt 的 <point_patch> 数一致）
            # 你也可以在 config 指定 grasp_num_group
            grasp_num_group = getattr(config, "grasp_num_group", None)
            if grasp_num_group is None:
                # 如果你仍然使用 point_backbone_config_name 的 yaml，就读它的 num_group
                point_bert_config_name = getattr(
                    config, "point_backbone_config_name", "PointTransformer_8192point_2layer"
                )
                point_bert_config_addr = os.path.join(
                    os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml"
                )
                point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
                grasp_num_group = int(point_bert_config.model.num_group)

            # GRASP residual feature dim (Cg) 必须告诉我们；最稳是你在 config 里显式写 grasp_feat_dim
            Cg = getattr(config, "grasp_feat_dim", None)
            if Cg is None:
                raise ValueError(
                    "config.grasp_feat_dim is required when point_backbone='GRASP' "
                    "(it is the feature dim produced by grasp_model.res_enc)."
                )

            # dense->ME coords 的量化参数：coord_range 与你的点云 normalize 对齐
            # 如果你的点云在 [-1,1]（常见 unit sphere），用 "sphere"
            coord_range = getattr(config, "grasp_coord_range", "sphere")  # "sphere" or "unit"
            grid_size = int(getattr(config, "grasp_grid_size", 256))
            use_fps = bool(getattr(config, "grasp_use_fps", True))

            grasp_args = GraspBackboneArgs(
                num_group=grasp_num_group,
                grid_size=grid_size,
                coord_range=coord_range,
                use_fps=use_fps,
                add_pos=True,
                cls_token=True,
            )

            self.point_backbone = GraspTokenBackbone(grasp_model=grasp_model, Cg=Cg, args=grasp_args)

            # backbone / projector 配置
            self.point_backbone_config = {
                "point_cloud_dim": 3,                 # 这里 GRASP 只用 xyz 做 coords
                "backbone_output_dim": Cg,            # projector input dim
                "project_output_dim": self.config.hidden_size,
                "point_token_len": grasp_num_group + 1,  # cls + G
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": 0,
                "use_max_pool": False,
            }

            logger.info(
                f"[GRASP] num_group={grasp_num_group}, token_len={self.point_backbone_config['point_token_len']}, "
                f"grid_size={grid_size}, coord_range={coord_range}, Cg={Cg}"
            )

            # Adapter: GRASP token dim != PointBERT trans_dim，所以这里要用 Cg
            try:
                from pointllm.model.transform_neck3d import TransformNeck3D
                neck_in_dim = Cg
                self.point_backbone.transform_neck3d = TransformNeck3D(in_dim=neck_in_dim)
                logger.info(f"[Adapter] transform_neck3d attached for GRASP: {self.point_backbone.transform_neck3d}")
            except Exception as e:
                logger.error(f"[Adapter] Failed to init TransformNeck3D module for GRASP: {e}")
                self.point_backbone.transform_neck3d = None
        # ========== projector 相关 ==========
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(
            f"Use {self.point_backbone_config['projection_hidden_layer']} "
            f"projection hiddent layers."
        )

        if self.point_backbone_config["projection_hidden_layer"] > 0:
            # 多层 MLP projector: Linear + GELU ... + Linear
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(
                    nn.Linear(
                        last_dim,
                        self.point_backbone_config["projection_hidden_dim"][i],
                    )
                )
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(
                nn.Linear(last_dim, self.point_backbone_config["project_output_dim"])
            )
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(
                f"Each layer with {point_bert_config.model.projection_hidden_dim} "
                f"hidden units."
            )
        else:
            # 单层 Linear projector
            self.point_proj = nn.Linear(
                backbone_output_dim,
                self.point_backbone_config["project_output_dim"],
            )

        logger.info(
            f"Point projector output dim: {self.point_backbone_config['project_output_dim']}."
        )

        self.fix_pointnet = getattr(config, "fix_pointnet", False)
        self.fix_llm = False

    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        self.point_backbone.load_checkpoint(
            self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        point_backbone = getattr(self, "point_backbone", None)
        point_backbone_config = getattr(self, "point_backbone_config", None)

        if (
            point_backbone is not None
            and (input_ids.shape[1] != 1 or self.training)
            and point_clouds is not None
        ):
            # 1) 只在 no_grad 里算 backbone 的原始特征
            if isinstance(point_clouds, list):
                raw_point_features = []
                with torch.no_grad() if self.fix_pointnet else nullcontext():
                    if self.fix_pointnet:
                        self.point_backbone.eval()
                    for point_cloud in point_clouds:
                        feat = self.point_backbone(point_cloud.unsqueeze(0))[0]  # [N, C]
                        raw_point_features.append(feat)
            else:
                with torch.no_grad() if self.fix_pointnet else nullcontext():
                    if self.fix_pointnet:
                        self.point_backbone.eval()
                    raw_point_features = self.point_backbone(point_clouds)  # [B, N, C] or [B, C]

            # 2) 在有梯度的区域内应用 adapter（如果存在）
            def apply_adapter(feat):
                neck = getattr(self.point_backbone, "transform_neck3d", None)
                if neck is not None:
                    return neck(feat)
                return feat

            if isinstance(point_clouds, list):
                point_features = [apply_adapter(f) for f in raw_point_features]
                # 3) 再过 projector
                point_features = [self.point_proj(f) for f in point_features]
            else:
                point_features = apply_adapter(raw_point_features)
                point_features = self.point_proj(point_features)

            dummy_point_features = torch.zeros(
                point_backbone_config["point_token_len"],
                self.point_backbone_config["backbone_output_dim"],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_point_features = self.point_proj(dummy_point_features)

            new_input_embeds = []
            cur_point_idx = 0
            for cur_input_ids, cur_input_embeds in zip(
                input_ids, inputs_embeds
            ):  # input_ids: B, L; input_embeds: B, L, C
                if (cur_input_ids == point_backbone_config["point_patch_token"]).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0.0 * dummy_point_features).sum()  # do nothing
                    new_input_embeds.append(cur_input_embeds)
                    cur_point_idx += 1
                    continue
                cur_point_features = point_features[cur_point_idx].to(device=cur_input_embeds.device)
                num_patches = cur_point_features.shape[0]  # number of point tokens
                if point_backbone_config["mm_use_point_start_end"]:
                    if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (
                        cur_input_ids == point_backbone_config["point_end_token"]
                    ).sum():
                        raise ValueError(
                            "The number of point start tokens and point end tokens should be the same."
                        )
                    point_start_tokens = torch.where(
                        cur_input_ids == point_backbone_config["point_start_token"]
                    )[0]
                    for point_start_token_pos in point_start_tokens:
                        if (
                            cur_input_ids[point_start_token_pos + num_patches + 1]
                            != point_backbone_config["point_end_token"]
                        ):
                            raise ValueError(
                                "The point end token should follow the point start token."
                            )
                        if (
                            orig_embeds_params is not None
                        ):  # will not update the original embeddings except for POINT_START/END
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:point_start_token_pos].detach(),
                                    cur_input_embeds[point_start_token_pos : point_start_token_pos + 1],
                                    cur_point_features,
                                    cur_input_embeds[
                                        point_start_token_pos
                                        + num_patches
                                        + 1 : point_start_token_pos
                                        + num_patches
                                        + 2
                                    ],
                                    cur_input_embeds[
                                        point_start_token_pos + num_patches + 2 :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: point_start_token_pos + 1],
                                    cur_point_features,
                                    cur_input_embeds[
                                        point_start_token_pos + num_patches + 1 :
                                    ],
                                ),
                                dim=0,
                            )
                        cur_point_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    if (
                        cur_input_ids == point_backbone_config["point_patch_token"]
                    ).sum() != num_patches:
                        raise ValueError(
                            "The number of point patch tokens should be the same as the number of point patches."
                        )
                    masked_indices = torch.where(
                        cur_input_ids == point_backbone_config["point_patch_token"]
                    )[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(
                            mask_index_start,
                            mask_index_start + num_patches,
                            device=masked_indices.device,
                            dtype=masked_indices.dtype,
                        )
                    ).any():
                        raise ValueError(
                            "The point patch tokens should be consecutive."
                        )
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_point_features,
                                cur_input_embeds[mask_index_start + num_patches :].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_point_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_point_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(PointLLMLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PointLLMLlamaForCausalLM(LlamaForCausalLM):
    config_class = PointLLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PointLLMLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    # ===== 新增：加载 TransformNeck3D adapter 权重 =====
    def load_point_adapter(self, adapter_ckpt: Optional[str] = None):
        # 1) 确定 ckpt 路径：显式传入优先，否则用 config 里的
        adapter_ckpt = adapter_ckpt or getattr(self.config, "point_adapter_ckpt", None)
        if not adapter_ckpt:
            logger.info("[Adapter] No adapter_ckpt provided. Skip loading TransformNeck3D adapter.")
            return

        neck = getattr(self.model.point_backbone, "transform_neck3d", None)
        if neck is None:
            logger.warning("[Adapter] transform_neck3d is None on point_backbone; cannot load adapter.")
            return

        logger.info(f"[Adapter] Loading adapter from: {adapter_ckpt}")
        # Use weights_only=False for compatibility with PyTorch 2.6+
        ckpt = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)

        # 2) 取出真正的 state_dict
        if isinstance(ckpt, dict) and "adapter" in ckpt:
            sd = ckpt["adapter"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        # 3) 清理前缀：最终 key 要变成 norm.weight / down.weight / up.weight ...
        clean_sd = {}
        for k, v in sd.items():
            k = k.replace("module.", "").replace("student.", "")
            if k.startswith("transform_neck3d."):
                k = k[len("transform_neck3d."):]
            if k.startswith("neck."):
                k = k[len("neck."):]
            clean_sd[k] = v

        missing, unexpected = neck.load_state_dict(clean_sd, strict=False)
        logger.info(
            f"[Adapter] Loaded into TransformNeck3D. missing={len(missing)}, unexpected={len(unexpected)}"
        )

        # 小 debug，可以确认不是 meta tensor
        try:
            logger.info(f"[Adapter] neck.norm.weight.mean() = {neck.norm.weight.mean().item():.6f}")
        except Exception:
            pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,  # * control whether to return past_key_values
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            point_clouds=point_clouds,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs

    def initialize_tokenizer_point_backbone_config_wo_embedding(self, tokenizer):
        # * called when stage2 or inference or inference without pre-training, assume tokenizer has point tokens
        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = (
            point_backbone_config["mm_use_point_start_end"]
        ) = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN

        tokenizer.add_tokens([default_point_patch_token], special_tokens=True)

        # * assert tokenizer has the default_point_patch_token
        point_backbone_config["default_point_patch_token"] = default_point_patch_token
        point_backbone_config["point_patch_token"] = tokenizer.convert_tokens_to_ids(
            [default_point_patch_token]
        )[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            tokenizer.add_tokens(
                [default_point_start_token, default_point_end_token],
                special_tokens=True,
            )

            point_backbone_config[
                "default_point_start_token"
            ] = default_point_start_token
            point_backbone_config["default_point_end_token"] = default_point_end_token

            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_start_token]
            )[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_end_token]
            )[0]

    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = (
            point_backbone_config["mm_use_point_start_end"]
        ) = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config["default_point_patch_token"] = default_point_patch_token
        tokenizer.add_tokens(
            [default_point_patch_token], special_tokens=True
        )  # no need to update embed since it will be replaced
        self.resize_token_embeddings(
            len(tokenizer)
        )  # resize_token_embeddings will make the tokens trainable again
        point_backbone_config["point_patch_token"] = tokenizer.convert_tokens_to_ids(
            [default_point_patch_token]
        )[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config[
                "default_point_start_token"
            ] = default_point_start_token
            point_backbone_config["default_point_end_token"] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens(
                [default_point_start_token, default_point_end_token], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_start_token]
            )[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids(
                [default_point_end_token]
            )[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [
                        self.get_input_embeddings()
                        .weight.data.clone()
                        .to(device=device)
                    ]  # only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters():  # the llm head
                        p.requires_grad = False
                    print(
                        f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable."
                    )
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print(
                        "Setting output embeddings and all input embeddings trainable."
                    )

    def measure_complexity(self, tokenizer, device=None, logger=None):
        """
        Measure and log model complexity metrics including:
        - Model size (MB/GB)
        - Trainable parameters (millions/billions)
        - kMACs (kilo Multiply-Accumulate operations)
        - Inference time (ms per point cloud)
        
        Args:
            tokenizer: Tokenizer instance for creating input_ids
            device: Device to run on (auto-detect if None)
            logger: Logger instance (uses module logger if None)
        
        Returns:
            Dictionary with all metrics
        """
        from pointllm.eval.model_stats import log_model_complexity_simple
        return log_model_complexity_simple(self, tokenizer, device=device, logger=logger)


AutoConfig.register("pointllm", PointLLMConfig)
AutoModelForCausalLM.register(PointLLMConfig, PointLLMLlamaForCausalLM)
