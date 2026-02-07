# Copyright (c) 2010-2022, InterDigital
# All rights reserved.

# See LICENSE under the root folder.


# Geometric subtraction and point analysis in the GRASP-Net paper

import os, sys
import torch
import torch.nn as nn
import numpy as np
from pccai.models.modules.pointnet import PointNet

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../third_party/nndistance'))
import importlib.util
from pathlib import Path

def _load_nndmodule_from_third_party():
    here = Path(__file__).resolve()

    # 向上找 repo root，直到找到 third_party/nndistance/modules/nnd.py
    repo_root = None
    for p in [here.parent, *here.parents]:
        cand = p / "third_party" / "nndistance" / "modules" / "nnd.py"
        if cand.exists():
            repo_root = p
            nnd_py = cand
            break

    if repo_root is None:
        raise RuntimeError(
            f"[NND] Cannot find third_party/nndistance/modules/nnd.py from {here}"
        )

    spec = importlib.util.spec_from_file_location("third_party_nnd", str(nnd_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[NND] spec load failed for {nnd_py}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NNDModule

NNDModule = _load_nndmodule_from_third_party()


nndistance = NNDModule()

try:
    import faiss
    import faiss.contrib.torch_utils

    found_FAISS = True
except ModuleNotFoundError:
    found_FAISS = False


class PointResidualEncoder(nn.Module):

    def __init__(self, net_config, **kwargs):
        super(PointResidualEncoder, self).__init__()

        syntax = kwargs['syntax']
        self.phase = syntax.phase.lower()
        self.k = net_config['k']
        self.thres_dist = net_config['thres_dist']
        self.feat_gen = PointNet(net_config, syntax=syntax)  # feature generation, the point analysis part
        self.faiss = (net_config.get('faiss', False) or self.phase != 'train') and found_FAISS
        self.faiss_resource, self.faiss_gpu_index_flat = None, None
        self.faiss_exact_search = True

    def forward(self, x_orig, x_coarse):
        # If Minkowski coords (N,4) = [b,x,y,z], always use the batch version
        if (
                torch.is_tensor(x_orig) and torch.is_tensor(x_coarse)
                and x_orig.dim() == 2 and x_orig.size(1) == 4
                and x_coarse.dim() == 2 and x_coarse.size(1) == 4
        ):
            geo_res = self.geo_subtraction_batch(x_orig, x_coarse)
        else:
            geo_subtraction = self.geo_subtraction_batch if self.phase == 'train' else self.geo_subtraction
            geo_res = geo_subtraction(x_orig, x_coarse)

        feat = self.feat_gen(geo_res)
        return feat

    # This is to perform geometric subtraction for point clouds in a batch manner
    def geo_subtraction_batch(self, x_orig, x_coarse):

        geo_res = torch.zeros((x_coarse.shape[0], self.k, 3), device=x_coarse.device)

        # ✅ ensure batch id col is integer-like for reliable masking & batch_size
        x_orig_b = x_orig[:, 0].long()
        x_coarse_b = x_coarse[:, 0].long()

        # ✅ robust batch size (works even if batches are missing some ids)
        batch_size = int(torch.max(x_orig_b).item()) + 1
        tot = 0

        for pc_cnt in range(batch_size):

            if self.faiss == True:  # FAISS for kNN search in one shot
                x_coarse_cur = (x_coarse[x_coarse[:, 0] == pc_cnt][:, 1:]).float().contiguous()  # current coarse
                x_orig_cur = (x_orig[x_orig[:, 0] == pc_cnt][:, 1:]).float().contiguous()  # current full cloud
                if self.faiss_gpu_index_flat == None:
                    self.faiss_resource = faiss.StandardGpuResources()
                    self.faiss_gpu_index_flat = faiss.GpuIndexFlatL2(self.faiss_resource, 3)
                self.faiss_gpu_index_flat.add(x_orig_cur)
                _, I = self.faiss_gpu_index_flat.search(x_coarse_cur, self.k)  # actual search
                self.faiss_gpu_index_flat.reset()
                x_coarse_rep = x_coarse_cur.unsqueeze(1).repeat_interleave(self.k, dim=1)
                geo_res[tot: tot + x_coarse_cur.shape[0], :, :] = x_orig_cur[I] - x_coarse_rep

                # Outlier removal
                mask = torch.logical_or(
                    torch.max(geo_res[tot: tot + x_coarse_cur.shape[0], :, :], dim=2)[0] > self.thres_dist,
                    torch.min(geo_res[tot: tot + x_coarse_cur.shape[0], :, :], dim=2)[0] < -self.thres_dist
                )  # True is outlier
                I[mask] = I[:, 0].unsqueeze(-1).repeat_interleave(self.k, dim=1)[
                    mask]  # get the indices of the first NN
                geo_res[tot: tot + x_coarse_cur.shape[0], :, :][mask] = x_orig_cur[I[mask]] - x_coarse_rep[
                    mask]  # recompute the outlier distance
                tot += x_coarse_cur.shape[0]

            else:  # nndistance for sequential nearest-neighbor search
                x_coarse_cur = (x_coarse[x_coarse[:, 0] == pc_cnt][:, 1:]).float().unsqueeze(
                    0).contiguous()  # current coarse
                x_orig_cur = (x_orig[x_orig[:, 0] == pc_cnt][:, 1:]).float().unsqueeze(
                    0).contiguous()  # current full cloud
                for nn_cnt in range(self.k):
                    if x_orig_cur.shape[1] > 0:
                        _, _, idx_coarse, _ = nndistance(x_coarse_cur, x_orig_cur)  # compute nearest neighbor
                        geo_res[tot: tot + x_coarse_cur.shape[1], nn_cnt, :] = x_orig_cur.squeeze(0)[
                                                                                   idx_coarse] - x_coarse_cur.squeeze(
                            0)  # residual in delta xyz
                        mask = torch.logical_and(
                            torch.max(geo_res[tot: tot + x_coarse_cur.shape[1], nn_cnt, :], dim=1)[
                                0] <= self.thres_dist,
                            torch.min(geo_res[tot: tot + x_coarse_cur.shape[1], nn_cnt, :], dim=1)[
                                0] >= -self.thres_dist
                        )  # False is outlier

                        seq_outlier = torch.arange(
                            tot, x_coarse_cur.shape[1] + tot,
                            device=mask.device
                        )[torch.logical_not(mask)]

                        seq_outlier = torch.arange(
                            tot, x_coarse_cur.shape[1] + tot,
                            device=geo_res.device
                        )[~mask]

                        geo_res[seq_outlier, nn_cnt, :] = geo_res[seq_outlier, nn_cnt - 1, :]

                        idx_coarse = idx_coarse[mask.unsqueeze(0)]  # keep in-range

                        keep = torch.ones(x_orig_cur.shape[1], dtype=torch.bool, device=x_orig_cur.device)
                        keep[idx_coarse.squeeze(0)] = False
                        x_orig_cur = x_orig_cur[:, keep, :]  # still [1, M, 3]

                    else:  # if there is no point left behind, replicate the last one
                        geo_res[tot: tot + x_coarse_cur.shape[1], nn_cnt:, :] = \
                            geo_res[tot: tot + x_coarse_cur.shape[1], nn_cnt - 1, :].unsqueeze(1)
                        break
                tot += x_coarse_cur.shape[1]
        return geo_res

    # This is to perform geometric subtraction for a point cloud, used during inference
    def geo_subtraction(self, x_orig, x_coarse):
        geo_res = torch.zeros(size=(x_coarse.shape[1], self.k, 3), device=x_coarse.device)
        x_orig, x_coarse = x_orig.squeeze(0), x_coarse.squeeze(0)
        self.faiss_resource = faiss.StandardGpuResources()

        # Perform kNN search
        if self.faiss_exact_search:  # exact search
            self.faiss_gpu_index_flat = faiss.GpuIndexFlatL2(self.faiss_resource, 3)
            self.faiss_gpu_index_flat.add(x_orig)
            _, I = self.faiss_gpu_index_flat.search(x_coarse, self.k)  # search in one shot
        else:  # approximate search
            self.faiss_gpu_index_flat = faiss.GpuIndexIVFFlat(self.faiss_resource, 3,
                                                              4 * int(np.ceil(np.sqrt(x_orig.shape[0]))),
                                                              faiss.METRIC_L2)
            self.faiss_gpu_index_flat.train(x_orig)
            self.faiss_gpu_index_flat.add(x_orig)
            I = torch.zeros(x_coarse.shape[0], self.k, device=x_coarse.device, dtype=torch.long)  # initialize the index
            max_query = 2 ** 16
            n_times = int(np.ceil(x_coarse.shape[0] / max_query))
            for cnt in range(n_times):  # search by batch due to limitation of GpuIndexIVFFlat
                slc = slice(cnt * max_query, x_coarse.shape[0] if cnt == n_times - 1 else (cnt + 1) * max_query - 1)
                I[slc, :] = self.faiss_gpu_index_flat.search(x_coarse[slc, :], self.k)[1]

        self.faiss_gpu_index_flat.reset()
        x_coarse_rep = x_coarse.unsqueeze(1).repeat_interleave(self.k, dim=1)
        geo_res = x_orig[I] - x_coarse_rep

        # Outlier removal
        mask = torch.logical_not(torch.logical_and(
            torch.max(geo_res, dim=2)[0] <= self.thres_dist,
            torch.min(geo_res, dim=2)[0] >= -self.thres_dist
        ))  # True is outlier
        I[mask] = I[:, 0].unsqueeze(-1).repeat_interleave(self.k, dim=1)[mask]  # get the indices of the first NN
        geo_res[mask] = x_orig[I[mask]] - x_coarse_rep[mask]  # recompute the outlier distance
        del I, x_coarse_rep, x_orig, x_coarse, mask
        return geo_res