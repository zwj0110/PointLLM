#!/usr/bin/env python3
# pip install numpy open3d

import numpy as np
import open3d as o3d
import argparse
import os

def read_point_cloud(file_name):
    if os.path.splitext(file_name)[1].lower() == '.ply':
        pcd = o3d.io.read_point_cloud(file_name)
    elif os.path.splitext(file_name)[1].lower() == '.bin':
        xyz = np.fromfile(file_name, dtype=np.float32).reshape((-1, 4))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    else:
        raise ValueError("Only support .ply or .bin")
    return pcd

def normalize_to_unit(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    p = np.asarray(pcd.points, dtype=np.float32)
    p = p - p.mean(axis=0, keepdims=True)
    s = np.max(np.linalg.norm(p, axis=1))
    if s > 0:
        p = p / s
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(p)
    return out

def compute_bbox_extent(pcd: o3d.geometry.PointCloud) -> float:
    bbox = pcd.get_axis_aligned_bounding_box()
    ext = bbox.get_extent()
    return float(np.max(ext))

def main(opt):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=opt.window_name, height=opt.window_height, width=opt.window_width)

    # ---- load ----
    pcd1 = read_point_cloud(opt.file_name)
    if pcd1.is_empty():
        raise ValueError("pcd1 is empty")
    pcd1 = normalize_to_unit(pcd1)
    pcd1.paint_uniform_color(opt.color1)

    geos = [pcd1]

    if opt.file_name2 != '.':
        pcd2 = read_point_cloud(opt.file_name2)
        if pcd2.is_empty():
            raise ValueError("pcd2 is empty")
        pcd2 = normalize_to_unit(pcd2)
        pcd2.paint_uniform_color(opt.color2)

        # ---- place side-by-side ----
        gap = float(opt.gap)
        # 用 bbox extent 估一个合理间距：2*extent + gap
        extent = max(compute_bbox_extent(pcd1), compute_bbox_extent(pcd2))
        shift = 2.2 * extent + gap
        pcd1.translate([-shift / 2.0, 0, 0])
        pcd2.translate([ shift / 2.0, 0, 0])
        geos = [pcd1, pcd2]

    # ---- add geometry ----
    for g in geos:
        vis.add_geometry(g)

    # ---- render options ----
    opt_render = vis.get_render_option()
    opt_render.point_size = float(opt.point_size)
    opt_render.light_on = False
    opt_render.background_color = np.array([1,1,1] if opt.white_bg else [0,0,0])

    # ---- set view ----
    ctr = vis.get_view_control()
    if opt.view_file != '.':
        param = o3d.io.read_pinhole_camera_parameters(opt.view_file)
        ctr.convert_from_pinhole_camera_parameters(param)

    # ---- run or capture ----
    if opt.output_file != '.':
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(opt.output_file, do_render=True)
        print("Saved:", opt.output_file)
        vis.destroy_window()
        return

    # 交互：你转动视角时，两边一起变（同一个相机）
    vis.run()

    # 可选：退出时保存当前视角
    if opt.save_view_file != '.':
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(opt.save_view_file, param)
        print("Saved view:", opt.save_view_file)

    vis.destroy_window()

def add_options(parser):
    parser.add_argument('--file_name', type=str, required=True, help='Left point cloud (.ply/.bin)')
    parser.add_argument('--file_name2', type=str, default='.', help='Right point cloud (.ply/.bin). "." for single.')
    parser.add_argument('--output_file', type=str, default='.', help='If not ".", save screenshot and exit.')
    parser.add_argument('--view_file', type=str, default='.', help='Load camera view json if provided.')
    parser.add_argument('--save_view_file', type=str, default='.', help='Save camera view json when window closes.')

    parser.add_argument('--color1', type=float, nargs=3, default=[0.10, 0.35, 0.95], help='Left color RGB [0,1]')
    parser.add_argument('--color2', type=float, nargs=3, default=[0.10, 0.35, 0.95], help='Right color RGB [0,1]')

    parser.add_argument('--point_size', type=float, default=3.0, help='Point size (8192 pts: 2~4 recommended)')
    parser.add_argument('--white_bg', action='store_true', help='Use white background (default false=black)')
    parser.add_argument('--gap', type=float, default=0.15, help='Extra gap between clouds after normalization')

    parser.add_argument('--window_name', type=str, default='Point Cloud Compare', help='Window name.')
    parser.add_argument('--window_height', type=int, default=1400, help='Window height.')
    parser.add_argument('--window_width', type=int, default=2400, help='Window width.')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = add_options(parser)
    opt = parser.parse_args()
    main(opt)
