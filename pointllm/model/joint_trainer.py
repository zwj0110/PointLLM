# joint_trainer_adapter.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class JointFeatureAlignmentTrainer:
    """
    Teacher: 原始 PointTransformer（无 adapter），输入 original pts
    Student: PointTransformer + transform_neck3d（只训练 adapter），输入 compressed pts

    我们对齐的位置是：
        f_t = T_backbone(original_pts)            # projector 之前
        f_s = neck( S_backbone(compressed_pts) )  # projector 之前 + adapter 后
    然后在这个位置上做 feature loss。
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_dataset,
        val_dataset=None,
        lr=1e-4,
        batch_size=8,
        save_dir="./output",
        device=None,
        use_cosine=True,
        freeze_student_backbone=True,
    ):
        # --------- 设备 ---------
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --------- Teacher：完全冻结，eval 模式 ---------
        self.teacher = teacher_model.to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # --------- Student：只训练 adapter，backbone 可选是否冻结 ---------
        self.student = student_model.to(self.device)

        if freeze_student_backbone:
            for name, p in self.student.named_parameters():
                p.requires_grad = False

            # 只打开 adapter 参数
            if not hasattr(self.student, "transform_neck3d"):
                raise ValueError("student 模型上没有属性 transform_neck3d，请先挂上 adapter 再传进来。")
            for p in self.student.transform_neck3d.parameters():
                p.requires_grad = True

        # backbone 用 eval，adapter 用 train（避免 backbone dropout 干扰）
        self.student.eval()
        if hasattr(self.student, "transform_neck3d"):
            self.student.transform_neck3d.train()

        # --------- DataLoader ---------
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        # --------- 优化器：只针对 requires_grad=True 的参数 ---------
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("没有任何可训练参数，请检查 freeze_student_backbone 和 adapter 参数设置。")
        self.opt = torch.optim.Adam(trainable_params, lr=lr)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.use_cosine = use_cosine

    # =============== feature loss：MSE + (可选) Cosine ===============
    def feature_loss(self, f_s, f_t):
        """
        f_s, f_t: 形状相同，通常是 [B, N, C] 或 [B, C]
        L_feat = ||f_s - f_t||_2^2  +  λ * (1 - cos(f_s, f_t))
        """
        # 基础 MSE
        mse = F.mse_loss(f_s, f_t)

        if self.use_cosine:
            # 展平 batch 和 token 维度，做平均 cosine
            dim = f_s.dim()
            if dim > 2:
                # [B, N, C] -> [B*N, C]
                f_s_flat = f_s.reshape(-1, f_s.shape[-1])
                f_t_flat = f_t.reshape(-1, f_t.shape[-1])
            else:
                f_s_flat = f_s
                f_t_flat = f_t

            cos = 1 - F.cosine_similarity(f_s_flat, f_t_flat, dim=-1).mean()
            loss = mse + 0.1 * cos
        else:
            loss = mse

        return loss, mse

    # =============== 一个 helper：抽取 projector 之前的特征 ===============
    @torch.no_grad()
    def extract_backbone_feat(self, model, pts):
        """
        调用 PointTransformer 得到 projector 之前的特征。
        这里直接用 model(pts) 的输出，假设：
            - use_max_pool=False:  [B, N, C]
            - use_max_pool=True:   [B, C]
        跟 PointLLM 里 self.point_backbone(point_clouds) 的接口保持一致。
        """
        model.eval()
        feats = model(pts)  # 直接复用原 forward
        return feats

    # =============== 训练一个 epoch ===============
    def _train_one_epoch(self, epoch_idx):
        self.student.eval()
        if hasattr(self.student, "transform_neck3d"):
            self.student.transform_neck3d.train()

        total_loss = 0.0
        total_mse = 0.0

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {epoch_idx}")
        for batch in pbar:
            orig_pts = batch["original"].to(self.device)  # 原始点云
            comp_pts = batch["grasp"].to(self.device)     # 压缩点云

            # ---- Teacher feature: projector 之前 ----
            with torch.no_grad():
                f_t = self.extract_backbone_feat(self.teacher, orig_pts)  # [B, N, C] or [B, C]
                f_t = f_t.detach()

            # ---- Student backbone feature ----
            with torch.no_grad():
                f_s_backbone = self.extract_backbone_feat(self.student, comp_pts)  # projector 之前、adapter 之前

            # ---- Adapter：只对 student feature 做映射 ----
            if not hasattr(self.student, "transform_neck3d"):
                raise ValueError("student 上没有 transform_neck3d，无法应用 adapter。")
            f_s = self.student.transform_neck3d(f_s_backbone)

            # ---- 对齐特征 ----
            loss, mse = self.feature_loss(f_s, f_t)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total_loss += loss.item()
            total_mse += mse.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse.item():.4f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_mse = total_mse / len(self.train_loader)
        print(f"[Train] Epoch {epoch_idx}: avg loss={avg_loss:.6f}, avg mse={avg_mse:.6f}")
        return avg_loss, avg_mse

    # =============== 验证（可选） ===============
    @torch.no_grad()
    def _validate(self, epoch_idx):
        if self.val_loader is None:
            return None, None

        self.teacher.eval()
        self.student.eval()
        if hasattr(self.student, "transform_neck3d"):
            self.student.transform_neck3d.eval()

        total_loss = 0.0
        total_mse = 0.0

        pbar = tqdm(self.val_loader, desc=f"[Val]   Epoch {epoch_idx}")
        for batch in pbar:
            orig_pts = batch["original"].to(self.device)
            comp_pts = batch["grasp"].to(self.device)

            f_t = self.extract_backbone_feat(self.teacher, orig_pts).detach()
            f_s_backbone = self.extract_backbone_feat(self.student, comp_pts)
            f_s = self.student.transform_neck3d(f_s_backbone)

            loss, mse = self.feature_loss(f_s, f_t)
            total_loss += loss.item()
            total_mse += mse.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse.item():.4f}",
            })

        avg_loss = total_loss / len(self.val_loader)
        avg_mse = total_mse / len(self.val_loader)
        print(f"[Val]   Epoch {epoch_idx}: avg loss={avg_loss:.6f}, avg mse={avg_mse:.6f}")
        return avg_loss, avg_mse

    # =============== 主训练循环 ===============
    def train(self, num_epochs=20, save_interval=5):
        best_val_loss = None

        for epoch in range(1, num_epochs + 1):
            train_loss, _ = self._train_one_epoch(epoch)
            val_loss, _ = self._validate(epoch)

            # 保存周期性 checkpoint（只存 adapter 更方便之后加载）
            if (epoch % save_interval) == 0:
                adapter_state = self.student.transform_neck3d.state_dict()
                ckpt_path = os.path.join(self.save_dir, f"adapter_epoch{epoch}.pth")
                torch.save({"adapter": adapter_state}, ckpt_path)
                print(f"[Save] adapter checkpoint saved to: {ckpt_path}")

            # 简单的 best-val 记录（可选）
            if val_loss is not None:
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = os.path.join(self.save_dir, "adapter_best.pth")
                    torch.save({"adapter": self.student.transform_neck3d.state_dict()}, best_path)
                    print(f"[Save] best adapter updated: {best_path}")
