from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

# 注意：ResidualMLPAdapter 假设已经在同一目录下
from .adapters import ResidualMLPAdapter


# -----------------------------
# 1. 辅助工具函数 (移到类定义外部，避免循环导入)
# -----------------------------
def dense_points_to_me_coords(points, grid_size, coord_range="sphere"):
    """
    将密集点云 [B, N, 3] 转换为 ME 格式的坐标 [B*N, 4] (batch_index, x, y, z)
    """
    B, N, _ = points.shape
    device = points.device

    # 归一化/缩放坐标到 [0, grid_size]
    if coord_range == "sphere":
        # 假设原始点在 [-1, 1] 之间
        xyz = (points + 1.0) / 2.0 * grid_size
    else:
        # 假设原始点在 [0, 1] 之间
        xyz = points * grid_size

    xyz = xyz.long()  # 量化

    # 创建 Batch Index
    batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, 1)
    me_coords = torch.cat([batch_indices.float(), xyz.float()], dim=-1)  # [B, N, 4]
    return me_coords.view(-1, 4).int()


def sample_per_batch_fixed_G(xyz_all, feat_all, b_idx, B, G, use_fps=True):
    """
    从稀疏特征中为每个 Batch 提取固定数量 G 的特征点
    """
    device = feat_all.device
    C = feat_all.shape[-1]

    out_xyz = torch.zeros(B, G, 3, device=device)
    out_feat = torch.zeros(B, G, C, device=device)

    for b in range(B):
        mask = (b_idx == b)
        curr_xyz = xyz_all[mask]
        curr_feat = feat_all[mask]

        n_curr = curr_xyz.shape[0]
        if n_curr == 0:
            continue

        if use_fps and n_curr > G:
            # 简单的 FPS 逻辑（或者你可以换成更高效的实现）
            # 这里简化为随机采样，FPS 实现通常需要第三方库
            idx = torch.randperm(n_curr, device=device)[:G]
        elif n_curr > G:
            idx = torch.arange(G, device=device)
        else:
            # 点数不够，重复填充
            idx = torch.cat([torch.arange(n_curr), torch.zeros(G - n_curr)]).long().to(device)

        out_xyz[b] = curr_xyz[idx]
        out_feat[b] = curr_feat[idx]

    return out_xyz, out_feat


def chamfer_distance(pc1, pc2):
    # pc1: (B, N, 3), pc2: (B, M, 3)
    # 增加简单的维度检查，防止 batch 为空
    if pc2.shape[1] == 0:
        return torch.tensor(0.0, device=pc1.device, requires_grad=True)

    dist_sq = torch.cdist(pc1, pc2)
    dist1 = dist_sq.min(dim=-1)[0].mean(dim=-1)
    dist2 = dist_sq.min(dim=-2)[0].mean(dim=-1)
    return (dist1 + dist2).mean()


from dataclasses import dataclass


@dataclass
class GraspBackboneArgs:
    """
    GRASP Backbone 的配置参数类
    """
    # 核心参数
    num_group: int  # 最终输入到 LLM 的 Token 数量 (G)
    grid_size: int = 256  # MinkowskiEngine 量化时的体素栅格大小
    coord_range: str = "sphere"  # 坐标范围类型: "sphere" ([-1,1]) 或 "unit" ([0,1])

    # 逻辑开关
    use_fps: bool = True  # 是否使用最远点采样 (FPS) 来选择 Token 坐标
    add_pos: bool = True  # 是否为特征添加位置编码 (Positional Embedding)
    cls_token: bool = True  # 是否在序列开头添加一个全局分类 Token

    # 模型维度与 Adapter 配置
    pos_hidden: int = 128  # 位置编码 MLP 的隐藏层维度
    adapter_hidden: int = 256  # ResidualMLPAdapter 的隐藏层维度
    adapter_dropout: float = 0.0  # Adapter 的 Dropout 概率

    # 压缩逻辑配置
    use_grasp_enc_adapter: bool = True  # 在量化瓶颈前是否使用 Adapter
    use_grasp_dec_adapter: bool = True  # 在量化瓶颈后是否使用 Adapter
# -----------------------------
# 2. 主类定义
# -----------------------------
class GraspTokenBackbone(nn.Module):
    def __init__(self, grasp_model: nn.Module, Cg: int, args: GraspBackboneArgs):
        super().__init__()
        self.grasp = grasp_model
        self.Cg = int(Cg)
        self.args = args

        # Adapter 1 & 2
        self.grasp_enc_adapter = ResidualMLPAdapter(Cg,
                                                    args.adapter_hidden) if args.use_grasp_enc_adapter else nn.Identity()
        self.grasp_dec_adapter = ResidualMLPAdapter(Cg,
                                                    args.adapter_hidden) if args.use_grasp_dec_adapter else nn.Identity()

        if args.add_pos:
            self.pos_mlp = nn.Sequential(
                nn.Linear(3, args.pos_hidden),
                nn.GELU(),
                nn.Linear(args.pos_hidden, Cg)
            )

        if args.cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, Cg))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _encode_coarse_and_feat_sparse(self, coords):
        """核心：将 ME 坐标输入 GRASP 获取稀疏特征"""
        if ME is None:
            raise ImportError("MinkowskiEngine is required for GRASP backbone.")

        # 转换坐标为 ME.SparseTensor 期望的输入
        # 注意：x_coarse_deq 在这里通常作为特征输入，如果是几何压缩，初始特征往往是全 1 或坐标本身
        # 假设 grasp.res_enc 接受 (coords, features)
        b_idx = coords[:, 0].long()
        xyz_raw = coords[:, 1:].float()

        # 初始化特征 (全1，代表占据点)
        feats = torch.ones(coords.shape[0], 1, device=coords.device)
        stensor = ME.SparseTensor(features=feats, coordinates=coords)

        # 调用 GRASP 编码器 (根据你实际的 GRASP 模型接口修改)
        # 假设 self.grasp.res_enc 返回的是每个点的特征向量
        feat_all = self.grasp.res_enc(stensor)

        # 如果返回的是 SparseTensor，提取 features
        if hasattr(feat_all, "F"):
            feat_all = feat_all.F

        xyz_norm = xyz_raw / self.args.grid_size
        return xyz_norm, feat_all, b_idx

    def forward(self, points_dense, return_loss=True):
        B = points_dense.shape[0]

        # 1. 转换坐标
        coords = dense_points_to_me_coords(points_dense[:, :, :3], self.args.grid_size, self.args.coord_range)

        # 2. 获取稀疏特征
        xyz_all, feat_all, b_idx = self._encode_coarse_and_feat_sparse(coords)

        # 3. Enc Adapter
        feat_all = self.grasp_enc_adapter(feat_all)

        # 4. 模拟量化 (R 损失)
        if self.training:
            noise = (torch.rand_like(feat_all) - 0.5) * (1.0 / 256.0)
            feat_all = feat_all + noise
        R_loss = torch.mean(torch.abs(feat_all)) * 0.01

        # 5. Dec Adapter
        feat_all = self.grasp_dec_adapter(feat_all)

        # 6. 特征 Token 化 (固定长度 G)
        xyz_tok, feat_tok = sample_per_batch_fixed_G(
            xyz_all, feat_all, b_idx, B, self.args.num_group, self.args.use_fps
        )

        # 7. 重建损失 (D 损失)
        # 用采样后的点回传计算 Chamfer，保证 D_loss 能通过 xyz_tok 更新
        if return_loss:
            D_loss = chamfer_distance(points_dense[:, :, :3], xyz_tok)
        else:
            D_loss = torch.tensor(0.0, device=feat_all.device)

        # 8. 位置编码与 CLS Token
        if hasattr(self, 'pos_mlp'):
            feat_tok = feat_tok + self.pos_mlp(xyz_tok)

        if hasattr(self, 'cls_token'):
            cls = self.cls_token.expand(B, -1, -1)
            out = torch.cat([cls, feat_tok], dim=1)
        else:
            out = feat_tok

        if return_loss:
            return out, R_loss, D_loss
        return out