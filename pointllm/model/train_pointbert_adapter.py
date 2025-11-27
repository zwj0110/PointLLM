# train_pointbert_adapter_blocks.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointllm.utils import cfg_from_yaml_file
from pointllm.model import PointTransformer
from pointllm.model.transform_neck3d import TransformNeck3D
from pointllm.model.dataset import ModelNet40DistillDataset


def register_block_hooks(model, target_layers):
    """
    在指定的 block 上注册 forward hook，收集中间特征。
    返回：handle 列表 + 一个 dict( layer_id -> [B, C] pooled feature )
    """
    feats = {}

    def make_hook(layer_id):
        def hook(module, inp, out):
            x = out
            # 通常 block 输出是 [B, N, C]；做一个简单的 global max pool 变成 [B, C]
            if x.dim() == 3:
                x, _ = x.max(dim=1)
            feats[layer_id] = x
        return hook

    handles = []
    for lid, block in enumerate(model.blocks.blocks):
        if lid in target_layers:
            h = block.register_forward_hook(make_hook(lid))
            handles.append(h)

    return handles, feats


def main():
    # ========== 1. config & dataset ==========
    cfg = cfg_from_yaml_file("./configs/PointTransformer_8192point_2layer.yaml")

    num_pts = getattr(cfg.model, "num_points", 8192)
    point_dims = getattr(cfg.model, "point_dims", 3)  # 你 yaml 里应该是 6
    use_max_pool = getattr(cfg.model, "use_max_pool", True)

    train_set = ModelNet40DistillDataset(
        original_root="./data/modelnet40_train_all",
        compressed_root="./data/bench_graspnet_r01_train_all",
        split="train",
        num_points=num_pts,
        cache_npy=False,
        strict_match=False,
        normalize_unit_sphere=True
    )

    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=4, drop_last=True
    )

    print("Train set size:", len(train_set))
    print("Num batches per epoch:", len(train_loader))

    # ========== 2. teacher / student ==========
    teacher_ckpt = "./checkpoints/point_bert_v1.2.pt"
    student_ckpt = "./checkpoints/point_bert_v1.2.pt"

    # Teacher
    teacher = PointTransformer(cfg.model, use_max_pool=use_max_pool)
    if teacher_ckpt is not None:
        teacher.load_checkpoint(teacher_ckpt)
    else:
        print("[WARN] No teacher_ckpt provided, teacher is randomly initialized.")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Student
    student = PointTransformer(cfg.model, use_max_pool=use_max_pool)
    if student_ckpt is not None:
        student.load_checkpoint(student_ckpt)

    # ===== 在后两个 blocks 上挂 adapter =====
    num_layers = student.depth
    hidden_dim = student.trans_dim
    # 最后两个 block 的 index：num_layers-2, num_layers-1
    target_layers = [num_layers - 2, num_layers - 1]

    for layer_id, block in enumerate(student.blocks.blocks):
        if layer_id in target_layers:
            block.adapter = TransformNeck3D(
                in_dim=hidden_dim,
                hidden_dim=256,
                dropout=0.1,
                scale=1.0,
            )
            print(f"[Init] Attach adapter to block {layer_id}")

    # 只训练 adapter 参数
    for p in student.parameters():
        p.requires_grad = False
    for name, p in student.named_parameters():
        if "adapter" in name:
            p.requires_grad = True

    print("Trainable params in student:")
    for n, p in student.named_parameters():
        if p.requires_grad:
            print("  ", n)

    # ========== 3. device & optimizer ==========
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

    # ========== 4. checkpoint 目录 ==========
    save_dir = "./output_pointbert_r01_block_adapters"
    os.makedirs(save_dir, exist_ok=True)

    num_epochs = 10
    save_every = 2
    best_loss = float("inf")

    # ========== 5. 训练循环（block-level loss） ==========
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        student.train()
        total_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(train_loader, start=1):
            pts_orig = batch["original"].to(device).float()
            pts_comp = batch["grasp"].to(device).float()

            optimizer.zero_grad()

            # --- 先在 teacher 上注册最后两层 block 的 hook ---
            teacher_handles, teacher_feats = register_block_hooks(
                teacher, target_layers
            )
            # teacher 不需要梯度
            with torch.no_grad():
                _ = teacher(pts_orig)
            # 用完先把 hook 解绑
            for h in teacher_handles:
                h.remove()

            # --- 再在 student 上注册同样的 block hook ---
            student_handles, student_feats = register_block_hooks(
                student, target_layers
            )
            _ = student(pts_comp)
            for h in student_handles:
                h.remove()

            # --- 每个 block 结束都算一次 loss，然后平均 ---
            block_losses = []
            for lid in target_layers:
                ft_t = teacher_feats[lid]       # [B, C]
                ft_s = student_feats[lid]       # [B, C]
                block_losses.append(F.mse_loss(ft_s, ft_t.detach()))

            feat_loss = sum(block_losses) / len(block_losses)

            feat_loss.backward()
            optimizer.step()

            bs = pts_orig.size(0)
            total_loss += feat_loss.item() * bs
            total_samples += bs
            global_step += 1

            if step % 10 == 0:
                avg_step_loss = total_loss / total_samples
                print(
                    f"Epoch [{epoch}/{num_epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"feature_loss = {avg_step_loss:.6f}"
                )

        avg_loss = total_loss / total_samples
        print(f"==== Epoch {epoch} done, avg feature_loss = {avg_loss:.6f} ====")

        # 按间隔存 epoch ckpt
        if epoch % save_every == 0:
            ckpt_path = f"{save_dir}/student_adapter_epoch{epoch}.pth"
            torch.save(student.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # 更新 best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"{save_dir}/student_adapter_best.pth"
            torch.save(student.state_dict(), best_path)
            print(f"New best model saved to: {best_path} (loss={best_loss:.6f})")

    final_path = f"{save_dir}/student_adapter_final.pth"
    torch.save(student.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
