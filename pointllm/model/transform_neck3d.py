import torch
import torch.nn as nn

class TransformNeck3D(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1, scale=1.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = scale

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = self.drop(x)
        return residual + self.scale * x
