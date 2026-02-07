# Copyright (c) 2010-2022, InterDigital
# All rights reserved. 

# See LICENSE under the root folder.


import math
import torch
import sys
import os

class PccLossBase:
    """A base class of rate-distortion loss computation for point cloud compression."""

    def __init__(self, loss_args, syntax):
        self.alpha = loss_args['alpha']
        self.beta = loss_args['beta']
        self.hetero = syntax.hetero
        self.phase = syntax.phase


    @staticmethod
    def bpp_loss(loss_out, likelihoods, count):
        """Compute the rate loss with the likelihoods."""

        bpp_loss = 0
        for k, v in likelihoods.items():
            if v is not None:
                loss = torch.log(v).sum() / (-math.log(2) * count)
                bpp_loss += loss
                loss_out[f'bpp_loss_{k}'] = loss.unsqueeze(0)
        loss_out['bpp_loss'] = bpp_loss.unsqueeze(0)


    def xyz_loss(self, **kwargs):
        """Needs to implement the xyz_loss"""

        raise NotImplementedError()


    def loss(self, **kwargs):
        """Needs to implement the overall loss. Can be R-D loss for lossy compression, or rate-only loss for lossless compression."""

        raise NotImplementedError()

import torch
import torch.nn.functional as F

class RDLambdaMSELoss(PccLossBase):
    """
    总损失: loss = R + beta * D + alpha * MSE_feat

    - R: 用 likelihoods 算 bpp（你 repo 里已有 bpp_loss）
    - D: pc_rec 和 pc_gt 的几何误差（这里先用 MSE，若点数不一致建议改 Chamfer）
    - MSE_feat: 学生/老师特征蒸馏 MSE（可选）
    """

    def __init__(self, loss_args, syntax):
        super().__init__(loss_args, syntax)
        # 允许 loss_args 里显式传 lambda / alpha 覆盖
        if "lambda" in loss_args:
            self.beta = float(loss_args["lambda"])
        if "alpha" in loss_args:
            self.alpha = float(loss_args["alpha"])

        self.distortion = loss_args.get("distortion", "mse")  # "mse" or "l1"
        self.rate_key = loss_args.get("rate_key", "likelihoods")  # out[rate_key]
        self.pc_gt_key = loss_args.get("pc_gt_key", "coords")     # data[pc_gt_key] or out[pc_gt_key]
        self.pc_rec_key = loss_args.get("pc_rec_key", "pc_rec")   # out[pc_rec_key] or out["x_hat"]
        self.feat_teacher_key = loss_args.get("feat_teacher_key", "feat_teacher")
        self.feat_student_key = loss_args.get("feat_student_key", "feat_student")

    def _get_pc_gt(self, data, out):
        if isinstance(out, dict) and self.pc_gt_key in out:
            return out[self.pc_gt_key]
        if isinstance(data, dict) and self.pc_gt_key in data:
            return data[self.pc_gt_key]
        if isinstance(data, dict) and "points" in data:
            return data["points"]
        raise KeyError(f"Cannot find gt point cloud in out/data. Tried keys: {self.pc_gt_key}, points")

    def _get_pc_rec(self, out):
        if self.pc_rec_key in out:
            return out[self.pc_rec_key]
        if "x_hat" in out:
            return out["x_hat"]
        if "pc_hat" in out:
            return out["pc_hat"]
        raise KeyError(f"Cannot find reconstructed point cloud in out. Tried keys: {self.pc_rec_key}, x_hat, pc_hat")

    def xyz_loss(self, pc_gt, pc_rec):
        # 这里假设 shape 一致：(B,N,3) 或 (N,3)
        if self.distortion == "l1":
            return F.l1_loss(pc_rec, pc_gt)
        return F.mse_loss(pc_rec, pc_gt)

    def loss(self, data, out):
        """
        返回 dict:
          - loss
          - bpp_loss / bpp_loss_xxx
          - dist_loss
          - mse_feat_loss (若启用)
        """
        loss_out = {}

        # ---------- Rate: R ----------
        likelihoods = out.get(self.rate_key, None)
        # count: 用每个样本点数 or 总点数都行，只要一致；这里用 “总点数”
        pc_gt = self._get_pc_gt(data, out)
        if pc_gt.dim() == 3:
            count = pc_gt.shape[0] * pc_gt.shape[1]
        else:
            count = pc_gt.shape[0]

        if likelihoods is not None:
            self.bpp_loss(loss_out, likelihoods, count)
            R = loss_out["bpp_loss"].squeeze(0)
        else:
            # 允许你某些阶段只做 D / MSE_feat
            R = pc_gt.new_tensor(0.0)

        # ---------- Distortion: D ----------
        pc_rec = self._get_pc_rec(out)
        D = self.xyz_loss(pc_gt, pc_rec)
        loss_out["dist_loss"] = D.unsqueeze(0)

        # ---------- Feature MSE: alpha * MSE ----------
        mse_feat = pc_gt.new_tensor(0.0)
        if self.alpha > 0:
            if (self.feat_student_key in out) and (self.feat_teacher_key in out):
                fs = out[self.feat_student_key]
                ft = out[self.feat_teacher_key]
                mse_feat = F.mse_loss(fs, ft)
            else:
                raise KeyError(
                    f"alpha>0 but out missing feat tensors: {self.feat_student_key} / {self.feat_teacher_key}"
                )
        loss_out["mse_feat_loss"] = mse_feat.unsqueeze(0)

        # ---------- Total ----------
        total = R + float(self.beta) * D + float(self.alpha) * mse_feat
        loss_out["loss"] = total.unsqueeze(0)
        return loss_out
