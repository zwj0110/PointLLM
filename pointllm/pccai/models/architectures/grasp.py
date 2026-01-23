import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import MinkowskiEngine as ME

from pccai.models.modules.get_modules import get_module_class
from pccai.models.utils_sparse import scale_sparse_tensor_batch, sort_sparse_tensor_with_dir

# 保持原有的 import 路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))
from entropy_model import EntropyBottleneck


class GeoResCompression(nn.Module):
    def __init__(self, net_config, syntax):
        super(GeoResCompression, self).__init__()

        # 基础参数加载
        self.dus = net_config.get('dus', 1)
        self.scaling_ratio = net_config['scaling_ratio']
        self.eb_channel = net_config['entropy_bottleneck']  # 应该为 8
        self.entropy_bottleneck = EntropyBottleneck(self.eb_channel)
        self.thres_dist = np.ceil((1 / self.scaling_ratio) * 0.65) if self.scaling_ratio < 0.5 else 1

        self.point_mul = net_config.get('point_mul', 5)
        self.skip_mode = net_config.get('skip_mode', False)
        if syntax.phase.lower() == 'train':
            self.noise = net_config.get('noise', -1)

        net_config['res_enc']['k'] = net_config['res_dec']['num_points'] = self.point_mul
        net_config['res_enc']['thres_dist'] = self.thres_dist
        net_config['res_dec']['dims'][0] = net_config['vox_dec']['dims'][-1]

        # 模块实例化
        self.res_dec = get_module_class(net_config['res_dec']['model'], False)(net_config['res_dec'], syntax=syntax)
        self.vox_dec = get_module_class(net_config['vox_dec']['model'], False)(net_config['vox_dec'], syntax=syntax)
        self.pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)
        if self.skip_mode == False:
            self.vox_enc = get_module_class(net_config['vox_enc']['model'], False)(net_config['vox_enc'], syntax=syntax)
            self.res_enc = get_module_class(net_config['res_enc']['model'], False)(net_config['res_enc'], syntax=syntax)

    def forward(self, input_pc, return_loss=True):
        """
        修改后的 Forward：支持 [B, N, 3] 输入，输出 [B, 513, 8]
        """
        device = input_pc.device

        # 1. 适配 PointLLM：将 [B, N, 3] 转为 ME.SparseTensor
        # 注意：这里不再使用原代码诡异的 cumsum 逻辑，改用标准的 Batch Index 构造
        if input_pc.dim() == 3:
            batch_size, num_points, _ = input_pc.shape
            # 坐标放大 64 倍以确保在 Minkowski 整数网格中有足够的体素
            coords_scaled = (input_pc[:, :, :3] * 64.0).int()

            # 构造带 Batch Index 的坐标 [B*N, 4]
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).repeat(1, num_points, 1)
            me_coords = torch.cat([batch_indices, coords_scaled], dim=-1).view(-1, 4)

            x = ME.SparseTensor(
                features=torch.ones(me_coords.shape[0], 1, device=device),
                coordinates=me_coords,
                device=device)
        else:
            x = input_pc  # 兼容处理
            batch_size = int(x.C[:, 0].max().item() + 1)

        # 2. 几何分析层 (GraspNet 核心)
        with torch.no_grad():
            x_coarse = scale_sparse_tensor_batch(x, factor=self.scaling_ratio)
            x_coarse = sort_sparse_tensor_with_dir(x_coarse)
            x_coarse_deq = torch.hstack((x_coarse.C[:, 0:1], (x_coarse.C[:, 1:].float() / self.scaling_ratio)))

        # 3. 提取特征 (Enhancement Layer)
        if self.skip_mode == False:
            feat = self.res_enc(x.C.float(), x_coarse_deq.float())
            x_feat = ME.SparseTensor(
                features=feat,
                coordinate_manager=x_coarse.coordinate_manager,
                coordinate_map_key=x_coarse.coordinate_map_key)
            y = self.vox_enc(x_feat)
            y_q, likelihood = get_likelihood(self.entropy_bottleneck, y)
        else:
            y_q, likelihood = x, torch.tensor(1.0, device=device)

        # --- [核心修改：稀疏到定长的对齐逻辑] ---
        # y_q.F 是所有 batch 压扁后的特征 [Total_Voxels, 8]
        # y_q.C 是对应的坐标 [Total_Voxels, 4]
        y_f, y_c = y_q.F, y_q.C
        target_n = 513
        target_dim = self.eb_channel  # 应该为 8

        # 结果容器 [B, 513, 8]
        hat_f_seq = torch.zeros((batch_size, target_n, target_dim), device=device)

        # 按 Batch 分发特征
        for b in range(batch_size):
            mask = (y_c[:, 0] == b)
            batch_feat = y_f[mask]

            if batch_feat.shape[0] > 0:
                # 使用插值确保每个 batch 都有 513 个 Token
                # 哪怕 Grasp 实际上只压缩出了 100 个点，插值也会让它变成 513 个
                if batch_feat.dim() == 1: batch_feat = batch_feat.unsqueeze(0)
                feat_input = batch_feat.unsqueeze(0).transpose(1, 2)  # [1, 8, N]
                upsampled = F.interpolate(feat_input, size=target_n, mode='linear', align_corners=False)
                hat_f_seq[b] = upsampled.squeeze(0).transpose(0, 1)

        # 4. 返回符合 PointLLM 预期的格式
        if return_loss:
            # 计算比特率损失 (Rate Loss)
            if isinstance(likelihood, dict):
                l_val = likelihood.get('feats', torch.tensor(1.0, device=device))
            else:
                l_val = likelihood
            R_loss = -torch.log2(l_val + 1e-6).mean()
            return hat_f_seq, R_loss, torch.tensor(0.0, device=device)

        return hat_f_seq


# 保持原有的 get_likelihood 函数不变
def get_likelihood(entropy_bottleneck, data):
    data_F, likelihood = entropy_bottleneck(data.F, quantize_mode="noise")
    data_Q = ME.SparseTensor(
        features=data_F,
        coordinate_map_key=data.coordinate_map_key,
        coordinate_manager=data.coordinate_manager,
        device=data.device)
    return data_Q, likelihood