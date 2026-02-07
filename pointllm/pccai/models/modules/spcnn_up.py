# Copyright (c) 2010-2022, InterDigital
# All rights reserved.
#
# See LICENSE under the root folder.

# Upsample with sparse CNN

import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2')
)
from data_utils import isin
from autoencoder import InceptionResNet, make_layer

# ✅ Adapter
from .adapters import MinkowskiAdapter


def make_sparse_up_block(in_dim, hidden_dim, out_dim, doLastRelu):
    """
    Make an up-sampling block based on InceptionResNet
    """
    layers = [
        ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3,
        ),
        ME.MinkowskiReLU(inplace=True),

        ME.MinkowskiConvolution(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        ),
        ME.MinkowskiReLU(inplace=True),
        MinkowskiAdapter(channels=out_dim),

        make_layer(
            block=InceptionResNet,
            block_layers=3,
            channels=out_dim,
        ),
    ]

    if doLastRelu:
        layers.append(ME.MinkowskiReLU(inplace=True))

    return nn.Sequential(*layers)


class SparseCnnUp1(nn.Module):
    """
    Up-sample once
    """

    def __init__(self, net_config, **kwargs):
        super().__init__()

        dims = net_config["dims"]

        self.up_block0 = make_sparse_up_block(
            dims[0], dims[1], dims[1], doLastRelu=False
        )


        self.pruning = ME.MinkowskiPruning()

    def forward(self, y1, gt_pc):
        out = self.up_block0(y1)
        # out = apply_conv1d_adapter(out, self.adapter1)
        out = self.prune_voxel(out, gt_pc.C)
        return out

    def prune_voxel(self, coarse_voxels, refined_voxels):
        mask = isin(coarse_voxels.C, refined_voxels)
        return self.pruning(coarse_voxels, mask.to(coarse_voxels.device))


class SparseCnnUp2(nn.Module):
    """
    Up-sample twice
    """

    def __init__(self, net_config, **kwargs):
        super().__init__()

        dims = net_config["dims"]

        # ---------- first up ----------
        self.up_block1 = make_sparse_up_block(
            dims[0], dims[1], dims[1], doLastRelu=True
        )
        # self.adapter1 = ResidualConv1dAdapter(channels=dims[1])

        # ---------- second up ----------
        self.up_block2 = make_sparse_up_block(
            dims[1], dims[2], dims[2], doLastRelu=False
        )
        # self.adapter2 = ResidualConv1dAdapter(channels=dims[2])

        self.pruning = ME.MinkowskiPruning()
        self.pool = ME.MinkowskiMaxPooling(
            kernel_size=2, stride=2, dimension=3
        )

    def forward(self, y1, gt_pc):
        # ----- first up -----
        out = self.up_block1(y1)
        # out = apply_conv1d_adapter(out, self.adapter1)

        y2_C = self.pool(gt_pc)
        out = SparseCnnUp1.prune_voxel(self, out, y2_C.C)

        # ----- second up -----
        out = self.up_block2(out)
        # out = apply_conv1d_adapter(out, self.adapter2)
        out = SparseCnnUp1.prune_voxel(self, out, gt_pc.C)

        return out
