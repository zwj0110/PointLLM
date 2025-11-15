from typing import Dict
import torch
from .modelnet_dir_dataset import ModelNet40Dir

class ModelNet40DirCompat(ModelNet40Dir):
    """
    目录直读的兼容版本，返回与 .dat 加载流程一致的字典：
      {
        "point_clouds": Tensor (N, C),  # C=3/6（默认 use_color=True → 6 通道）
        "labels":       LongTensor 标量,
        "label_names":  str,
        "indice":       LongTensor 标量
      }
    规则：
      - 先得到 xyz (N,3)
      - if use_color=True: 拼接一份同形状的全 0 → (N,6) 以匹配 in_chans=6
      - 如需 height，可再扩展，但你当前权重是 6 通道，默认不开启
    """
    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color

    def __getitem__(self, idx: int) -> Dict[str, object]:
        base = super().__getitem__(idx)  # {'points': Tensor(N,3), 'label': int, 'path': str}
        pts = base["points"]             # (N,3) float32 tensor
        if self.use_color:
            pts = torch.cat([pts, torch.zeros_like(pts)], dim=-1)  # (N,6)

        label = int(base["label"])
        label_name = self.categories[label]

        return {
            "point_clouds": pts,                               # (N,6) or (N,3)
            "labels": torch.tensor(label, dtype=torch.long),   # scalar
            "label_names": label_name,                         # str
            "indice": torch.tensor(idx, dtype=torch.long),     # scalar
        }
