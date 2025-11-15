import os
import trimesh

# ———— 修改这里 ———
SRC_ROOT = '/Users/zhengwenjie/Downloads/ModelNet40'       # 原始 OFF 文件根目录
DST_ROOT = 'ModelNet40_ply'   # 输出 PLY 文件根目录
# ————————————————

for class_name in os.listdir(SRC_ROOT):
    src_test_dir = os.path.join(SRC_ROOT, class_name, 'test')
    if not os.path.isdir(src_test_dir):
        continue

    dst_test_dir = os.path.join(DST_ROOT, 'test')
    os.makedirs(dst_test_dir, exist_ok=True)

    for fname in os.listdir(src_test_dir):
        if not fname.lower().endswith('.off'):
            continue

        src_path = os.path.join(src_test_dir, fname)
        dst_fname = fname[:-4] + '.ply'
        dst_path = os.path.join(dst_test_dir, dst_fname)

        # 加载 OFF 并导出 PLY（保持顶点 & 面片）
        mesh = trimesh.load(src_path, force='mesh', process=False)
        mesh.export(dst_path)

        print(f'✅ {src_path} → {dst_path}')
