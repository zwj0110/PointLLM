import os
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from collections import OrderedDict

from .dvae import Group
from .dvae import Encoder
from .logger import print_log
from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from ..transform_neck3d import TransformNeck3D  # ★ 你的 adapter 类


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE: scale factor
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # ★ Block 级别 adapter（训练时就是挂在这里）
        self.adapter: nn.Module | None = None

    def forward(self, x):
        # 标准 Transformer block
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # 如果有 adapter，则再走一层
        if self.adapter is not None:
            x = self.adapter(x)  # [B, G+1, C]

        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder without hierarchical structure."""

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()
        self.config = config

        self.use_max_pool = use_max_pool

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.point_dims = config.point_dims

        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(
            encoder_channel=self.encoder_dims,
            point_input_dims=self.point_dims
        )
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # （可选）全局 neck adapter：如果你不想用，可以不理它
        self.transform_neck3d: nn.Module | None = None

    # ========== Adapter 初始化：和你训练脚本保持一致 ==========
    # ========== Adapter 初始化：和你训练脚本保持一致 ==========
    def init_adapters(
            self,
            start_layer: int | None = None,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            scale: float = 1.0,
    ):
        """
        在最后两个 Block 上挂 TransformNeck3D（或者从 start_layer 开始一直到最后）。

        默认：从 depth-2 开始，也就是只给最后两个 block 加 adapter。
        """
        if start_layer is None:
            # depth 可能是 12，就会从 10 开始 → block 10 和 block 11
            start_layer = max(self.depth - 2, 0)

        # ★ 目标层集合：start_layer, ..., depth-1
        target_layers = list(range(start_layer, self.depth))

        # ★ 保证 adapter 在和 backbone 相同的 device + dtype 上（cuda / mps / cpu + fp32 / bf16）
        base_param = next(self.parameters())
        device = base_param.device
        dtype = base_param.dtype

        for layer_id, block in enumerate(self.blocks.blocks):
            if layer_id in target_layers:
                if block.adapter is None:
                    adapter = TransformNeck3D(
                        in_dim=self.trans_dim,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        scale=scale,
                    )
                    # 关键：同时对齐 device 和 dtype
                    adapter = adapter.to(device=device, dtype=dtype)
                    block.adapter = adapter

                print_log(
                    f"[PointTransformer] Attach adapter to block {layer_id}",
                    logger="Transformer",
                )



    # ========== 加载 adapter 参数（只加载带 'adapter' 的 key） ==========
    def load_adapter_checkpoint(self, adapter_ckpt_path: str, only_adapter: bool = True):
        """
        从训练好的 PointBERT student checkpoint 里加载 adapter 参数。

        Args:
            adapter_ckpt_path: 训练脚本保存的 student ckpt 路径
            only_adapter:      True 时仅加载包含 'adapter' 的参数
        """
        if not os.path.isfile(adapter_ckpt_path):
            print_log(
                f"[PointTransformer] adapter checkpoint not found: {adapter_ckpt_path}",
                logger="Transformer",
            )
            return

        ckpt = torch.load(adapter_ckpt_path, map_location="cpu")

        # 兼容直接 state_dict 或 {'state_dict': ...} 两种格式
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if only_adapter:
            state_dict = {k: v for k, v in state_dict.items() if "adapter" in k}

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log("Adapter missing_keys", logger="Transformer")
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger="Transformer",
            )
        if incompatible.unexpected_keys:
            print_log("Adapter unexpected_keys", logger="Transformer")
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger="Transformer",
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            print_log(
                f"[PointTransformer] Adapter weights successfully loaded from {adapter_ckpt_path}",
                logger="Transformer",
            )

    # ========== 原始 PointBERT backbone 权重加载 ==========
    def load_checkpoint(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = OrderedDict()

        # 从 transformer_q.* 中提取 backbone 参数
        for k, v in ckpt['state_dict'].items():
            if k.startswith('transformer_q.'):
                new_k = k.replace('transformer_q.', '')
                state_dict[new_k] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger='Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Transformer'
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            print_log(
                f"PointBERT's weights are successfully loaded from {bert_ckpt_path}",
                logger='Transformer'
            )

    # ========== 前向 ==========
    def forward(self, pts):
        # divide the point cloud
        neighborhood, center = self.group_divider(pts)
        # encode cloud blocks
        group_input_tokens = self.encoder(neighborhood)           # [B, G, C_enc]
        group_input_tokens = self.reduce_dim(group_input_tokens)  # [B, G, trans_dim]

        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  # [B, 1, trans_dim]
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)      # [B, 1, trans_dim]

        # pos embedding
        pos_tokens = self.pos_embed(center)  # [B, G, trans_dim]

        # concat
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # [B, G+1, trans_dim]
        pos = torch.cat((cls_pos, pos_tokens), dim=1)           # [B, G+1, trans_dim]

        # transformer
        x = self.blocks(x, pos)  # [B, G+1, trans_dim]
        x = self.norm(x)         # [B, G+1, trans_dim]

        # 可选的全局 neck adapter（不是你这次训练那套，可以先不启用）
        if self.transform_neck3d is not None:
            x = self.transform_neck3d(x)

        if not self.use_max_pool:
            # 返回 token 级特征给 PointLLM 做 point_token_len 对齐
            return x  # [B, G+1, trans_dim]

        # 用 cls 和 token max-pool 拼接成全局特征
        pooled = torch.cat(
            [x[:, 0], x[:, 1:].max(1)[0]],
            dim=-1
        ).unsqueeze(1)  # [B, 1, 2*trans_dim]

        return pooled
