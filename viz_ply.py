import open3d as o3d
import numpy as np

def main():
    # === 修改这里，指定你的 ply 文件路径 ===
    # data/converted_xyz_structured/guitar/test/guitar_0158.ply
    # data/bench_surface_dense_r01_new/guitar/test/guitar_0156.ply
    file_path = "data/test_datasets_sample/laptop/grasp_laptop_0150_r05.ply"

    # 读取点云
    pcd = o3d.io.read_point_cloud(file_path)

    if not pcd.has_points():
        raise ValueError("点云为空，请检查路径或文件内容")

    # 设置统一颜色（这里用黑色，可以改成白色 [1,1,1] 或灰色 [0.5,0.5,0.5]）
    pcd.paint_uniform_color([0, 0, 0])

    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PLY Viewer", width=1200, height=900)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])  # 白色背景
    render_opt.point_size = 2.0  # 点的大小

    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
