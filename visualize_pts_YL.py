#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 11:02:35 2025

@author: ying
"""

import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from pointllm.utils import *
from pointllm.data.utils import *

#save_path = 'pointllm/data/modelnet40_data/modelnet40_test_8192pts_fps.dat'
save_path = 'pointllm/data/modelnet40_data/modelnet40_test_8192pts_fps_reconstructed.dat'
with open(save_path, 'rb') as f:
    list_of_points, list_of_labels = pickle.load(f) # * ndarray of N, C: (8192, 6) (xyz and normals)

import matplotlib.pyplot as plt

# --- load your data (pick ONE of these) ---
# pts = np.load('points.npy')                 # if saved as N×3 .npy
# pts = np.loadtxt('points.txt')              # if whitespace/CSV text: x y z per line
# pts = np.genfromtxt('points.csv', delimiter=',')  # if CSV
# For demo, make some dummy points:
# pts = np.random.randn(8192, 3)

for i in range(len(list_of_points)):
    pts = list_of_points[i+20*i][:,:3]
    assert pts.shape[1] == 3, "Expect shape N×3 for (x,y,z)."
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.1, color='blue')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.1)
    
    # Make axes equal for true geometry
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    ax.set_box_aspect(ranges)  # (x, y, z) equal aspect
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Point Cloud (XYZ)")
    plt.show()