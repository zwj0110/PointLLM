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
student_ckpt = None

# ================================
# 2. Init teacher (freeze)
# ================================
teacher = PointTransformer(cfg.model, use_max_pool=True)
teacher.load_checkpoint(teacher_ckpt)

for p in teacher.parameters():
    p.requires_grad = False

# ================================
# 3. Init student (trainable)
# ================================
student = PointTransformer(cfg.model, use_max_pool=True)

if student_ckpt:
    student.load_checkpoint(student_ckpt)

# 插入你的 Transform-Neck3D
student.transform_neck3d = TransformNeck3D(in_dim=cfg.model.trans_dim)

# ================================
# 4. Load dataset (原始 + 压缩)
# ================================
num_pts = getattr(cfg.model, 'num_points', 8192)
train_set = ModelNet40DistillDataset(
    original_root="./data/modelnet40_train_all_8192",
    compressed_root="./data/bench_surface_dense_r04_train",
    split="train",
    num_points=num_pts,
    cache_npy=True,
    strict_match=True
)

val_set = ModelNet40DistillDataset(
    original_root="./data/modelnet40_test_all_8192",
    compressed_root="./data/bench_surface_dense_r04_test",
    split="test",
    num_points=num_pts,
    cache_npy=True,
    strict_match=True
)

# ================================
# 5. Train
# ================================
trainer = JointFeatureAlignmentTrainer(
    teacher_model=teacher,
    student_model=student,
    train_dataset=train_set,
    val_dataset=val_set,
    lr=3e-4,
    save_dir="./output_r04"
)

trainer.train(num_epochs=5, save_interval=1)
