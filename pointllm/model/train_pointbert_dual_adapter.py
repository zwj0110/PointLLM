# train_pointbert_adapter.py
# 只训练 PointBERT Block 内的 attn_adapter & ffn_adapter，用 feature distillation

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointllm.utils import cfg_from_yaml_file
from pointllm.model import PointTransformer
from pointllm.model.dataset import ModelNet40DistillDataset  # 你原来用的 Dataset


def extract_feat(model, pts):
    """
    从 PointTransformer 提取全局特征：
    - 如果返回 [B, 1, F]，则取 [:, 0, :]
    - 如果返回 [B, F]，直接用
    """
    x = model(pts)   # [B, 1, F] 或 [B, F]
    if x.dim() == 3:
        x = x[:, 0, :]
    return x


def main():
    # ========== 1. config & dataset ==========
    cfg = cfg_from_yaml_file("./configs/PointTransformer_8192point_2layer.yaml")
    num_pts = getattr(cfg.model, "num_points", 8192)

    train_set = ModelNet40DistillDataset(
        original_root="./data/modelnet40_train_all_8192",
        compressed_root="./data/bench_surface_dense_r02_train",
        split="train",
        num_points=num_pts,
        cache_npy=True,
        strict_match=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=32,          # M1 显存不大，爆显存就再减一点
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=False,       # MPS 下没必要 pin_memory
    )

    # ========== 2. teacher / student ==========
    teacher_ckpt = "./checkpoints/pointbert_pretrained_converted.pth"

    # Teacher: 冻结的原始 PointBERT（不带 adapter）
    teacher = PointTransformer(cfg.model, use_max_pool=True)
    teacher.load_checkpoint(teacher_ckpt)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Student: 结构相同，但在 Block 里挂 adapter
    student = PointTransformer(cfg.model, use_max_pool=True)
    student.load_checkpoint(teacher_ckpt)

    # ---------- 在 Block 内挂两个 adapter（attn_adapter & ffn_adapter） ----------
    num_layers = student.depth
    start_layer = num_layers // 2      # 后半层挂 adapter，和推理那边保持一致

    student.init_adapters(
        start_layer=start_layer,
        hidden_dim=256,                 # TransformNeck3D 的 bottleneck dim
        dropout=0.1,
        scale=0.1,                      # 训练时用 0.1 比较稳
    )

    # ---------- 只训练 adapter ----------
    for p in student.parameters():
        p.requires_grad = False
    for name, p in student.named_parameters():
        # attn_adapter.* 和 ffn_adapter.* 的名字里都带 "adapter"
        if "attn_adapter" in name or "ffn_adapter" in name:
            p.requires_grad = True

    print("Trainable params in student:")
    for n, p in student.named_parameters():
        if p.requires_grad:
            print("  ", n)

    # ========== 3. 设备选择（优先 MPS） ==========
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)
    teacher.to(device)
    student.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    # ========== 4. checkpoint 目录 & 记录 ==========
    save_dir = "./output_transformer_adapter_r02"
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 10
    save_every = 2
    best_loss = float("inf")

    # ========== 5. 训练循环 ==========
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader, start=1):
            # 你的 dataset: {'name', 'original', 'grasp'}
            pts_orig = batch["original"].to(device).float()  # 原始点云
            pts_comp = batch["grasp"].to(device).float()     # 压缩点云

            optimizer.zero_grad()

            # Teacher：原始点云（冻结，不反传）
            with torch.no_grad():
                feat_teacher = extract_feat(teacher, pts_orig)

            # Student：压缩点云（含 Block 内 adapters）
            feat_student = extract_feat(student, pts_comp)

            # ====== Loss：MSE( student_feat, teacher_feat ) ======
            # 即：让「压缩点云 + adapter」的全局特征去逼近
            # 「原始点云 + 冻结 PointBERT」的全局特征
            feat_loss = F.mse_loss(feat_student, feat_teacher.detach())

            feat_loss.backward()
            optimizer.step()

            bs = pts_orig.size(0)
            total_loss += feat_loss.item() * bs
            total_samples += bs
            global_step += 1

            # 小进度条：每 10 个 step 打一次
            if step % 10 == 0:
                avg_step_loss = total_loss / total_samples
                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"feature_loss = {avg_step_loss:.6f}"
                )

        # epoch 结束，算一个 epoch 平均 loss
        avg_loss = total_loss / total_samples
        print(f"==== Epoch {epoch} done, avg feature_loss = {avg_loss:.6f} ====")

        # 1）按间隔存一份 ckpt
        if epoch % save_every == 0:
            ckpt_path = f"{save_dir}/student_adapter_epoch{epoch}.pth"
            torch.save(student.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # 2）保存最优 loss 的一份
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"{save_dir}/student_adapter_best.pth"
            torch.save(student.state_dict(), best_path)
            print(f"New best model saved to: {best_path} (loss={best_loss:.6f})")

    # 最后再存一份 final
    final_path = f"{save_dir}/student_adapter_final.pth"
    torch.save(student.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
