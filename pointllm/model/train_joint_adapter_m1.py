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

teacher_ckpt = "./checkpoints/pointbert_pretrained_converted.pth"
student_ckpt = None  # 如果想从某个 student ckpt 继续训，可以填路径

use_max_pool = cfg.model.use_max_pool  # ⚠️ 一定要跟 PointLLM 里保持一致

# ================================
# 2. Init teacher (freeze)
# ================================
teacher = PointTransformer(cfg.model, use_max_pool=use_max_pool)
teacher.load_checkpoint(teacher_ckpt)
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
    compressed_root="./data/bench_surface_dense_r03_train",
    split="train",
    num_points=num_pts,
    cache_npy=True,
    strict_match=True,
)

val_set = ModelNet40DistillDataset(
    original_root="./data/modelnet40_test_all_8192",
    compressed_root="./data/bench_surface_dense_r03_test",
    split="test",
    num_points=num_pts,
    cache_npy=True,
    strict_match=True,
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
    save_dir="./output_r03_projector",
    use_cosine=True,
    freeze_student_backbone=True,  # 只训 adapter
)

trainer.train(num_epochs=10, save_interval=1)
