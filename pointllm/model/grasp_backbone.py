# pointllm/model/grasp_backbone.py
# Dense point cloud -> Minkowski coords -> GRASP-Net encoder tokens -> (B, G+1, Cg)

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except Exception as e:
    ME = None


# -----------------------------
# 1) Dense -> ME coords
# -----------------------------
@torch.no_grad()
def dense_points_to_me_coords(
    points_xyz: torch.Tensor,
    grid_size: int = 256,
    coord_range: str = "sphere",  # "sphere"([-1,1]) or "unit"([0,1])
) -> torch.Tensor:
    """
    Args:
        points_xyz: (B, N, 3) float
        grid_size: voxel resolution
        coord_range:
            - "sphere": points are in [-1, 1]
            - "unit": points are in [0, 1]
    Returns:
        coords: (M, 4) int32, columns: [b, x, y, z], unique per batch
    """
    assert points_xyz.dim() == 3 and points_xyz.size(-1) == 3
    B, N, _ = points_xyz.shape
    device = points_xyz.device

    xyz = points_xyz.contiguous()

    if coord_range == "sphere":
        xyz = (xyz + 1.0) * 0.5
    elif coord_range == "unit":
        pass
    else:
        raise ValueError(f"Unsupported coord_range={coord_range}. Use 'sphere' or 'unit'.")

    xyz = xyz.clamp(0.0, 1.0)

    # Quantize to integer grid
    q = torch.floor(xyz * (grid_size - 1) + 1e-6).to(torch.int32)  # (B,N,3)

    # Add batch index
    b = torch.arange(B, device=device, dtype=torch.int32).view(B, 1, 1).expand(B, N, 1)
    coords = torch.cat([b, q], dim=-1).reshape(-1, 4)  # (B*N,4)

    # Deduplicate per batch (unique coords required by MinkowskiEngine)
    G = int(grid_size)
    bb = coords[:, 0].to(torch.int64)
    x = coords[:, 1].to(torch.int64)
    y = coords[:, 2].to(torch.int64)
    z = coords[:, 3].to(torch.int64)
    key = bb * (G**3) + x + y * G + z * (G**2)

    # torch.unique(..., return_index=True) is available on newer torch; fallback if needed
    try:
        _, idx = torch.unique(key, sorted=False, return_inverse=False, return_counts=False, return_index=True)
        coords = coords[idx]
    except TypeError:
        # Fallback: sort then unique
        sorted_key, order = torch.sort(key)
        coords_sorted = coords[order]
        keep = torch.ones_like(sorted_key, dtype=torch.bool)
        keep[1:] = sorted_key[1:] != sorted_key[:-1]
        coords = coords_sorted[keep]

    return coords.to(torch.int32)


# -----------------------------
# 2) Simple FPS sampler
# -----------------------------
@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (N, 3) float
    return: (npoint,) long indices
    Simple O(N*npoint) FPS (CPU/GPU torch).
    """
    N = xyz.shape[0]
    if N <= npoint:
        # pad by repeating
        idx = torch.arange(N, device=xyz.device)
        if N < npoint:
            pad = idx[torch.randint(0, N, (npoint - N,), device=xyz.device)]
            idx = torch.cat([idx, pad], dim=0)
        return idx.to(torch.long)

    centroids = torch.empty((npoint,), device=xyz.device, dtype=torch.long)
    distance = torch.full((N,), float("inf"), device=xyz.device)
    farthest = torch.randint(0, N, (1,), device=xyz.device).item()

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.argmax(distance).item()
    return centroids


@torch.no_grad()
def sample_per_batch_fixed_G(
    xyz_all: torch.Tensor,   # (Nc_total, 3)
    feat_all: torch.Tensor,  # (Nc_total, Cg)
    batch_idx: torch.Tensor, # (Nc_total,)
    B: int,
    G: int,
    use_fps: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return:
        xyz_tok: (B, G, 3)
        feat_tok: (B, G, Cg)
    """
    device = xyz_all.device
    Cg = feat_all.shape[-1]

    xyz_out = torch.zeros((B, G, 3), device=device, dtype=xyz_all.dtype)
    feat_out = torch.zeros((B, G, Cg), device=device, dtype=feat_all.dtype)

    for b in range(B):
        mask = (batch_idx == b)
        xyz_b = xyz_all[mask]
        feat_b = feat_all[mask]
        if xyz_b.numel() == 0:
            continue

        if use_fps:
            idx = farthest_point_sample(xyz_b, G)
        else:
            N = xyz_b.shape[0]
            if N >= G:
                idx = torch.randperm(N, device=device)[:G]
            else:
                base = torch.arange(N, device=device)
                pad = base[torch.randint(0, N, (G - N,), device=device)]
                idx = torch.cat([base, pad], dim=0)

        xyz_out[b] = xyz_b[idx]
        feat_out[b] = feat_b[idx]

    return xyz_out, feat_out


# -----------------------------
# 3) GRASP -> token backbone
# -----------------------------
@dataclass
class GraspBackboneArgs:
    num_group: int                      # G from PointBERT config
    grid_size: int = 256                # voxel resolution for ME coords
    coord_range: str = "sphere"         # "sphere" or "unit"
    use_fps: bool = True
    add_pos: bool = True               # add xyz positional embedding
    cls_token: bool = True             # prepend cls token
    pos_hidden: int = 128              # xyz->Cg MLP hidden size


class GraspTokenBackbone(nn.Module):
    """
    Wrap a GRASP GeoResCompression model to produce fixed-length token sequence for PointLLM.
    Output: (B, G+1, Cg) if cls_token=True else (B, G, Cg)
    """

    def __init__(self, grasp_model: nn.Module, Cg: int, args: GraspBackboneArgs):
        super().__init__()
        if ME is None:
            raise ImportError("MinkowskiEngine is required for GraspTokenBackbone but not installed/importable.")
        self.grasp = grasp_model
        self.Cg = int(Cg)
        self.args = args

        if args.cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.Cg))
            # optional learned cls positional bias
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.Cg))
        else:
            self.cls_token = None
            self.cls_pos = None

        if args.add_pos:
            self.pos_mlp = nn.Sequential(
                nn.Linear(3, args.pos_hidden),
                nn.GELU(),
                nn.Linear(args.pos_hidden, self.Cg),
            )
        else:
            self.pos_mlp = None

    @torch.no_grad()
    def _build_sparse_from_coords(self, coords: torch.Tensor) -> "ME.SparseTensor":
        device = coords.device
        feats = torch.ones((coords.shape[0], 1), device=device, dtype=torch.float32)
        return ME.SparseTensor(features=feats, coordinates=coords, device=device)

    def _encode_coarse_and_feat_sparse(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode-only path (no entropy bottle / no decoder):
        Returns:
            xyz_all:  (Nc_total, 3) float
            feat_all: (Nc_total, Cg) float
            b_idx:    (Nc_total,) int64
        """
        # NOTE: We DO NOT call GeoResCompression.forward() because it mutates batch indices.
        from ..pccai.models.utils_sparse import scale_sparse_tensor_batch, sort_sparse_tensor_with_dir

        x = self._build_sparse_from_coords(coords)  # base layer sparse tensor

        # coarse quantization + dequant
        x_coarse = scale_sparse_tensor_batch(x, factor=self.grasp.scaling_ratio)
        x_coarse = sort_sparse_tensor_with_dir(x_coarse)

        b_idx = x_coarse.C[:, 0].to(torch.int64)  # (Nc_total,)
        xyz_all = (x_coarse.C[:, 1:].float() / self.grasp.scaling_ratio).contiguous()  # (Nc_total,3)

        # residual feature attached to coarse
        # res_enc API in your repo uses: feat = res_enc(x.C, x_coarse_deq)
        # where x_coarse_deq is (Nc_total,4) [b,xyz] float
        x_coarse_deq = torch.hstack([b_idx.to(xyz_all.dtype).view(-1, 1), xyz_all])  # (Nc_total,4)
        feat_all = self.grasp.res_enc(x.C, x_coarse_deq)  # (Nc_total, Cg)
        return xyz_all, feat_all, b_idx

    def forward(self, points_dense: torch.Tensor) -> torch.Tensor:
        if self.grasp is None:
            raise RuntimeError(
                "[GRASP] grasp model is None. You must build/load GeoResCompression "
                "and pass it into GraspTokenBackbone(grasp_model=...)."
            )

        """
        points_dense: (B, N, 3) or (B, N, 6). only xyz used for coords.
        returns: (B, G+1, Cg) or (B, G, Cg)
        """
        assert points_dense.dim() == 3 and points_dense.size(-1) >= 3
        B = points_dense.size(0)
        xyz = points_dense[:, :, :3]

        coords = dense_points_to_me_coords(
            xyz,
            grid_size=self.args.grid_size,
            coord_range=self.args.coord_range,
        )

        # Encode to coarse xyz + residual feat
        xyz_all, feat_all, b_idx = self._encode_coarse_and_feat_sparse(coords)

        # Sample fixed G tokens per batch
        xyz_tok, feat_tok = sample_per_batch_fixed_G(
            xyz_all=xyz_all,
            feat_all=feat_all,
            batch_idx=b_idx,
            B=B,
            G=self.args.num_group,
            use_fps=self.args.use_fps,
        )  # (B,G,3), (B,G,Cg)

        # Add xyz pos embedding if enabled
        if self.pos_mlp is not None:
            feat_tok = feat_tok + self.pos_mlp(xyz_tok)

        # Prepend cls token if enabled
        if self.args.cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            if self.cls_pos is not None:
                cls = cls + self.cls_pos.expand(B, -1, -1)
            out = torch.cat([cls, feat_tok], dim=1)  # (B,G+1,Cg)
        else:
            out = feat_tok  # (B,G,Cg)

        return out
