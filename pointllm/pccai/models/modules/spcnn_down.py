# Copyright (c) 2010-2022, InterDigital
# All rights reserved.
#
# See LICENSE under the root folder.

# Downsample with sparse CNN

import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

# PCGCv2 (keep if required by your project)
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2')
)
from autoencoder import InceptionResNet, make_layer

# ✅ Adapter (唯一新增依赖)
from .adapters import MinkowskiAdapter


def make_sparse_down_block(in_dim, hidden_dim, out_dim, doLastRelu=False):
    """
    Make a down-sampling block based on InceptionResNet
    """
    layers = [
        ME.MinkowskiConvolution(
            in_channels=in_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        ),
        ME.MinkowskiReLU(inplace=True),
        MinkowskiAdapter(channels=hidden_dim),

        ME.MinkowskiConvolution(
            in_channels=hidden_dim,
            out_channels=out_dim,
            kernel_size=2,
            stride=2,
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


class SparseCnnDown1(nn.Module):
    """
    Down-sample once
    """

    def __init__(self, net_config, **kwargs):
        super().__init__()

        dims = net_config["dims"]

        self.down_block0 = make_sparse_down_block(
            dims[0], dims[0], dims[1], doLastRelu=True
        )

        self.conv_last = ME.MinkowskiConvolution(
            in_channels=dims[1],
            out_channels=dims[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3,
        )

    def forward(self, x):
        x = self.down_block0(x)
        x = self.conv_last(x)
        return x


class SparseCnnDown2(nn.Module):
    """
    Down-sample twice
    """

    def __init__(self, net_config, **kwargs):
        super().__init__()

        dims = net_config["dims"]

        # ---------- first down ----------
        self.down_block2 = make_sparse_down_block(
            dims[0], dims[0], dims[1], doLastRelu=True
        )

        # ---------- second down ----------
        self.down_block1 = make_sparse_down_block(
            dims[1], dims[1], dims[2], doLastRelu=False
        )


    def forward(self, x):
        x = self.down_block2(x)
        # x = self.adapter2(x)

        x = self.down_block1(x)
        # x = self.adapter1(x)

        return x
