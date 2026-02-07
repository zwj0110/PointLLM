# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME

# If you need PCGCv2 path (keep if your project requires it)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/PCGCv2'))


class MinkowskiAdapter(nn.Module):
    """
    Implementation of the 'Purple Adapter' from the GRASP-Net diagram.
    Structure: SConv(1x1, C->C/8) -> ReLU -> SConv(1x1, C/8->C) -> Residual Add

    This replaces the previous generic MinkowskiAdapter.
    """

    def __init__(
            self,
            channels: int,  # Changed 'in_channels' to 'channels' to match previous API
            ratio: int = 8,  # Changed 'reduction_ratio' to 'ratio' to match previous API
            use_alpha: bool = True,  # Learnable gate
            alpha_init: float = 1.0,
            enabled: bool = True,  # Kept for compatibility, though usually True
    ):
        super().__init__()

        self.enabled = enabled

        # Ensure hidden dimension is at least 1
        # Diagram specifies C/8, so we use ratio=8 by default
        hidden_dim = max(1, channels // int(ratio))

        # 1. First SConv: C -> C/8, Kernel 1x1x1
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3,
        )

        # 2. ReLU
        self.act = ME.MinkowskiReLU(inplace=True)

        # 3. Second SConv: C/8 -> C, Kernel 1x1x1
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=hidden_dim,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3,
        )

        # Learnable gating parameter (alpha)
        # This allows the adapter to start with 0 influence and gradually turn on.
        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32))
        else:
            self.register_parameter("alpha", None)

        self._init_weights()

    def _init_weights(self):
        """
        Zero-initialize the second convolution.
        This ensures the adapter outputs near-zero values at the start,
        preserving the original pre-trained model behavior (Identity).
        """
        with torch.no_grad():
            if hasattr(self.conv2, "kernel") and self.conv2.kernel is not None:
                self.conv2.kernel.zero_()
            if hasattr(self.conv2, "bias") and self.conv2.bias is not None:
                self.conv2.bias.zero_()

    def forward(self, x):
        if not self.enabled:
            return x

        # Forward pass through the bottleneck
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        # Apply gating
        if self.alpha is not None:
            # Using tanh or sigmoid allows smooth gating from 0.
            # Original PurpleAdapter code used tanh here.
            gate = torch.tanh(self.alpha)
            out = out * gate

        # Residual Connection: X_out = X_in + Adapter(X_in)
        return x + out


class ResidualConv1dAdapter(nn.Module):
    """
    Dense residual adapter for token sequence [B, T, C].

    out = x + sigmoid(alpha) * f(x)
    Identity-ish at init via zero-init conv2 and alpha_init<0.
    """

    def __init__(
            self,
            channels: int,
            ratio: int = 8,
            use_alpha: bool = True,
            alpha_init: float = -4.0,
    ):
        super().__init__()
        mid = max(1, channels // int(ratio))

        self.conv1 = nn.Conv1d(channels, mid, kernel_size=1, bias=True)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(mid, channels, kernel_size=1, bias=True)

        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([float(alpha_init)], dtype=torch.float32))
        else:
            self.register_parameter("alpha", None)

        # zero-init last layer => f(x) ~ 0
        with torch.no_grad():
            self.conv2.weight.zero_()
            if self.conv2.bias is not None:
                self.conv2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        residual = x
        y = x.transpose(1, 2)  # [B, C, T]
        y = self.conv2(self.act(self.conv1(y)))
        y = y.transpose(1, 2)  # [B, T, C]

        if self.alpha is None:
            return residual + y

        gate = torch.sigmoid(self.alpha)
        return residual + y * gate


class ResidualMLPAdapter(nn.Module):
    """
    Dense residual MLP adapter.

    out = x + sigmoid(alpha) * f(x)
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = None,
            ratio: int = 8,
            use_alpha: bool = True,
            alpha_init: float = -4.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(1, in_dim // int(ratio))

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=True)

        if use_alpha:
            self.alpha = nn.Parameter(torch.tensor([float(alpha_init)], dtype=torch.float32))
        else:
            self.register_parameter("alpha", None)

        with torch.no_grad():
            self.fc2.weight.zero_()
            if self.fc2.bias is not None:
                self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(x)))
        if self.alpha is None:
            return x + out
        gate = torch.sigmoid(self.alpha)
        return x + out * gate