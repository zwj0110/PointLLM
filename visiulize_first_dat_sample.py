#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ———— Update this to your .dat file path ————
fp = "data/modelnet40_data_original/modelnet40_test_8192pts_fps_test1.dat"
# ————————————————————————————————————————

if not os.path.isfile(fp):
    raise FileNotFoundError(f"File not found: {fp}")

# 1. Load the pickled data
data = pickle.load(open(fp, 'rb'))
pcs_list = data[0]
pcs = np.array(pcs_list, dtype=np.float32) if isinstance(pcs_list, list) else pcs_list

# 2. Inspect shape
print(f"Point-cloud array shape: {pcs.shape}  (samples, points, dims)")

# 3. Extract the first sample
sample0 = pcs[0]
xyz  = sample0[:, :3]
dims = sample0.shape[1]

# 4. Visualize with Matplotlib (all points in black)
fig = plt.figure(figsize=(8, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("First Sample Point Cloud (XYZ Only)")

ax.scatter(
    xyz[:, 0],
    xyz[:, 1],
    xyz[:, 2],
    s=1,
    color='black'
)

ax.set_axis_off()
plt.show()
