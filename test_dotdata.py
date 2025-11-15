import pickle, numpy as np

dat = 'data/modelnet40_data/modelnet40_test_8192pts_fps.dat'  # 换成你的路径
pts_list, lbl_list = pickle.load(open(dat, 'rb'))

# 取前几个样本看坐标范围与半径
for i in range(3):
    pts = np.array(pts_list[i])   # (8192, 6) or (8192, 3/6)
    xyz = pts[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    print(f'#{i} min/mean/max xyz: {xyz.min():.3f}/{xyz.mean():.3f}/{xyz.max():.3f} '
          f' | max radius: {r.max():.3f}')