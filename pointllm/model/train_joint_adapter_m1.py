# train_adapter.py

import torch
from pointllm.model import PointTransformer
from .transform_neck3d import TransformNeck3D
from .joint_trainer import JointFeatureAlignmentTrainer
from .dataset import ModelNet40DistillDataset
from pointllm.utils import cfg_from_yaml_file

# ================================
# 1. Load config (model arch only)
# ================================
cfg = cfg_from_yaml_file("./configs/PointTransformer_8192point_2layer.yaml")

# ⚠️ 确保这里的 point_dims 和 ckpt 对齐
# point_bert_v1.2.pt 用的是 6 维（例如 xyz + 额外特征）
# 你的 yaml 里 model.point_dims 也应该是 6（你说已经改过了）
teacher_ckpt = "./checkpoints/point_bert_v1.2.pt"
student_ckpt = "./checkpoints/point_bert_v1.2.pt"  # 学生也从同一个 ckpt 起步

use_max_pool = cfg.model.use_max_pool  # ⚠️ 一定要跟 PointLLM 里保持一致
point_dims = cfg.model.point_dims      # 这里读出 6

# ================================
# 2. Init teacher (freeze)
# ================================
teacher = PointTransformer(cfg.model, use_max_pool=use_max_pool)
if teacher_ckpt is not None:
    teacher.load_checkpoint(teacher_ckpt)
else:
    print("[WARN] No teacher_ckpt provided, teacher is randomly initialized.")

for p in teacher.parameters():
    p.requires_grad = False

# ================================
# 3. Init student backbone + adapter
# ================================
student = PointTransformer(cfg.model, use_max_pool=use_max_pool)
if student_ckpt:
    student.load_checkpoint(student_ckpt)

# 在 backbone 输出之后加 adapter（projector 之前）
student.transform_neck3d = TransformNeck3D(in_dim=cfg.model.trans_dim)
print("[Init] Attached TransformNeck3D to student backbone.")

# ================================
# 4. Dataset: 原始 + 压缩
# ================================
num_pts = getattr(cfg.model, 'num_points', 8192)

train_set = ModelNet40DistillDataset(
    original_root="./data/modelnet40_train_all_8192",
    compressed_root="./data/bench_graspnet_r04_train_all",
    split="train",
    num_points=num_pts,
    cache_npy=True,
    strict_match=True,
    normalize_unit_sphere=True,
    point_dims=point_dims,   # ⭐ 关键：告诉 dataset 我们要 6 维
)

val_set = ModelNet40DistillDataset(
    original_root="./data/modelnet40_test_all_8192",
    compressed_root="./data/bench_graspnet_r04_test_all",
    split="test",
    num_points=num_pts,
    cache_npy=True,
    strict_match=True,
    normalize_unit_sphere=True,
    point_dims=point_dims,   # ⭐ 同上
)

# ================================
# 5. Train
# ================================
trainer = JointFeatureAlignmentTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataset=train_set,
    val_dataset=val_set,      # 如果没有 val_set，这里可以直接设为 None
    lr=3e-4,
    batch_size=8,
    save_dir="./output_r04_projector",
    use_cosine=True,
    freeze_student_backbone=True,  # 只训 adapter
)

trainer.train(num_epochs=10, save_interval=1)
