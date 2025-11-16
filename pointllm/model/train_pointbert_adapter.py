# train_pointbert_adapter.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointllm.utils import cfg_from_yaml_file
from pointllm.model import PointTransformer
from pointllm.model.transform_neck3d import TransformNeck3D
from pointllm.model.dataset import ModelNet40DistillDataset


def extract_feat(model, pts):
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
        train_set, batch_size=32, shuffle=True, num_workers=4, drop_last=True
    )

    # ========== 2. teacher / student ==========
    teacher_ckpt = "./checkpoints/pointbert_pretrained_converted.pth"
    teacher = PointTransformer(cfg.model, use_max_pool=True)
    teacher.load_checkpoint(teacher_ckpt)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student = PointTransformer(cfg.model, use_max_pool=True)
    student.load_checkpoint(teacher_ckpt)

    # 挂 adapter（方案2：后半层）
    num_layers = student.depth
    hidden_dim = student.trans_dim
    start_layer = num_layers // 2

    for layer_id, block in enumerate(student.blocks.blocks):
        if layer_id >= start_layer:
            block.adapter = TransformNeck3D(
                in_dim=hidden_dim,
                hidden_dim=256,
                dropout=0.1,
                scale=1.0,
            )
            print(f"Attach adapter to block {layer_id}")

    # 只训 adapter
    for p in student.parameters():
        p.requires_grad = False
    for name, p in student.named_parameters():
        if "adapter" in name:
            p.requires_grad = True

    print("Trainable params in student:")
    for n, p in student.named_parameters():
        if p.requires_grad:
            print("  ", n)

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

    # ========== 3. checkpoint 目录 & 记录 ==========
    save_dir = "./output_pointbert_r02"
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 10
    save_every = 2          # ✅ 每多少个 epoch 存一份
    best_loss = float("inf")

    # ========== 4. 训练循环 ==========
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader, start=1):
            # 你的 dataset 是 dict: {'name', 'original', 'grasp'}
            pts_orig = batch["original"].to(device).float()
            pts_comp = batch["grasp"].to(device).float()
            # name = batch["name"]  # 这个是字符串 list，不参与训练

            optimizer.zero_grad()

            with torch.no_grad():
                feat_teacher = extract_feat(teacher, pts_orig)

            feat_student = extract_feat(student, pts_comp)

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

        # ✅ 1）按间隔存一份 ckpt
        if epoch % save_every == 0:
            ckpt_path = f"{save_dir}/student_adapter_epoch{epoch}.pth"
            torch.save(student.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # ✅ 2）同时保存最优 loss 的一份
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"{save_dir}/student_adapter_best.pth"
            torch.save(student.state_dict(), best_path)
            print(f"New best model saved to: {best_path} (loss={best_loss:.6f})")

    # （可选）最后再存一份 final
    final_path = f"{save_dir}/student_adapter_final.pth"
    torch.save(student.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
