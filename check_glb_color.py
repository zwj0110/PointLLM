#!/usr/bin/env python3
import trimesh
import numpy as np
import os

def check_glb(fp):
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"File not found: {fp}")

    # 加载为三角网格
    mesh = trimesh.load(fp, force='mesh')
    visual = mesh.visual

    # 1) 顶点色（vertex colors）
    has_vc = hasattr(visual, 'vertex_colors') \
             and visual.vertex_colors is not None \
             and len(visual.vertex_colors) == len(mesh.vertices)

    # 2) 嵌入或外部贴图
    material = getattr(visual, 'material', None)
    has_texture = False
    tex_shape = None
    if material is not None and getattr(material, 'image', None) is not None:
        img = material.image
        tex = np.asarray(img)
        has_texture = True
        tex_shape = tex.shape  # (H, W, C)

    # 3) UV 坐标
    has_uv = hasattr(visual, 'uv') and visual.uv is not None and len(visual.uv) > 0

    print("=== GLB Color Check ===")
    print(f"Vertex colors present? {has_vc}")
    print(f"UV coords present?      {has_uv}")
    print(f"Embedded texture?      {has_texture}")
    if has_texture:
        print(f"Texture resolution:    {tex_shape[1]}×{tex_shape[0]} (W×H), channels={tex_shape[2]}")

    # 结论
    if has_vc:
        print("\n结论: 模型带有顶点色 (vertex colors).")
    elif has_texture and has_uv:
        print("\n结论: 模型带有基于 UV 的贴图 (texture)，可以生成带颜色的点云。")
    else:
        print("\n结论: 模型不包含颜色信息 (既无顶点色也无贴图)。")

if __name__ == "__main__":
    fp = "/Users/zhengwenjie/hf-objaverse-v1/glbs/000-003/fb42332b3f5e491cb0c4b5ba7ed6f374.glb"
    check_glb(fp)
