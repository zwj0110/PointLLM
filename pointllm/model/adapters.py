import torch
import torch.nn as nn

class ResidualMLPAdapter(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [N, C] or [B, G, C]
        shortcut = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return self.norm(x + shortcut)