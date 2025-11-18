import os
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from collections import OrderedDict

from .dvae import Group
from .dvae import Encoder
from .logger import print_log
from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from ..transform_neck3d import TransformNeck3D  # ★ Adapter 类


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
    """
    单个 Transformer Block

    结构（pre-norm）：

        x ---------------> + -----------------> x
           |            /                       ^
           |           /                        |
        LayerNorm   Multi-head Attn
                       |
                     Adapter_attn (可选)

        x ---------------> + -----------------> x
           |            /
           |           /
        LayerNorm       FFN
                         |
                     Adapter_ffn (可选)

    对应你发的图：Adapter 挂在 attention / FFN 之后，
    再和原始残差相加。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

        # ★ 两个子层上的 Adapter（注意：默认是 None，外面再 init）
        self.attn_adapter: nn.Module | None = None
        self.ffn_adapter: nn.Module | None = None

    def forward(self, x):
        # ===== Self-Attention 子层 =====
        attn_out = self.attn(self.norm1(x))          # [B, N, C]
        # 图上：Multi-head attention -> FF layer(投影) -> Adapter -> Residual
        if self.attn_adapter is not None:
            # TransformNeck3D 内部已经带 residual：y = x + scale * f(x)
            attn_out = self.attn_adapter(attn_out)   # [B, N, C]
        x = x + self.drop_path(attn_out)

        # ===== FFN 子层 =====
        ffn_out = self.mlp(self.norm2(x))            # [B, N, C]
        if self.ffn_adapter is not None:
            ffn_out = self.ffn_adapter(ffn_out)      # [B, N, C]
        x = x + self.drop_path(ffn_out)

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

        # ★ 全局 projector adapter（PointBERT student 里用）
        self.transform_neck3d: nn.Module | None = None

    # ========== Block Adapter 初始化：在每个 Block 的 attn / ffn 子层挂 adapter ==========
    def init_adapters(
            self,
            start_layer: int | None = None,
            hidden_dim: int = 256,
            dropout: float = 0.1,
            scale: float = 0.1,
    ):
        """
        在指定层之后的 Block 上挂 TransformNeck3D 作为 Houlsby-style adapter。

        默认从 depth // 2 开始挂（后半部分）。
        """
        if start_layer is None:
            start_layer = self.depth // 2

        device = next(self.parameters()).device

        for layer_id, block in enumerate(self.blocks.blocks):
            if layer_id >= start_layer:
                if block.attn_adapter is None:
                    block.attn_adapter = TransformNeck3D(
                        in_dim=self.trans_dim,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        scale=scale,
                    ).to(device)
                if block.ffn_adapter is None:
                    block.ffn_adapter = TransformNeck3D(
                        in_dim=self.trans_dim,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        scale=scale,
                    ).to(device)

                print_log(
                    f"[PointTransformer] Attach adapters to block {layer_id} "
                    f"(attn_adapter & ffn_adapter).",
                    logger="Transformer",
                )

    # ========== Projector Adapter 初始化 ==========
    def init_projector_adapter(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        scale: float = 0.1,
    ):
        """
        在 pooled 全局特征后挂一个 TransformNeck3D，作为 projector adapter。

        - 如果 use_max_pool=True，则 pooled 维度是 2 * trans_dim
        - 否则返回 token 级特征时，可以把 in_dim 设为 trans_dim（如果你想在 token 上做）
        """
        device = next(self.parameters()).device
        if self.use_max_pool:
            in_dim = self.trans_dim * 2
        else:
            in_dim = self.trans_dim

        self.transform_neck3d = TransformNeck3D(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            scale=scale,
        ).to(device)

        print_log(
            "[Student] Attach global TransformNeck3D as projector adapter.",
            logger="Transformer",
        )

    # ========== 加载 adapter 参数（Block + Projector） ==========
    def load_adapter_checkpoint(self, adapter_ckpt_path: str, only_adapter: bool = True):
        """
        从训练好的 PointBERT student checkpoint 里加载 adapter 参数。

        Args:
            adapter_ckpt_path: 训练脚本保存的 student ckpt 路径
            only_adapter:      True 时仅加载 adapter 参数
                               （blocks.*.attn_adapter / ffn_adapter / transform_neck3d）
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
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if ("adapter" in k) or ("transform_neck3d" in k)
            }

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

        # ★ 全局 projector adapter：对 pooled 特征做一个 TransformNeck3D
        if not self.use_max_pool:
            # 返回 token 级特征给 PointLLM 做 point_token_len 对齐
            if self.transform_neck3d is not None:
                x = self.transform_neck3d(x)
            return x  # [B, G+1, trans_dim]

        # 用 cls 和 token max-pool 拼接成全局特征
        pooled = torch.cat(
            [x[:, 0], x[:, 1:].max(1)[0]],
            dim=-1
        ).unsqueeze(1)  # [B, 1, 2*trans_dim]

        if self.transform_neck3d is not None:
            pooled = self.transform_neck3d(pooled)  # [B, 1, 2*trans_dim]

        return pooled
