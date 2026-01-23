# Copyright (c) 2010-2022, InterDigital
# All rights reserved.

# See LICENSE under the root folder.

# A ModelNet data loader (modified to support reading exported PLYs)

import os
import os.path
import glob
import re
import numpy as np
import pickle

import torch
import torch.utils.data as data
from torch_geometric.transforms.sample_points import SamplePoints
from torch_geometric.datasets.modelnet import ModelNet
from pccai.utils.convert_octree import OctreeOrganizer
import pccai.utils.logger as logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path_default = os.path.abspath(os.path.join(BASE_DIR, '../../datasets/modelnet/'))  # OFF 的默认路径


# ===== Utils =====
def gen_rotate():
    rot = np.eye(3, dtype='float32')
    rot[0, 0] *= np.random.randint(0, 2) * 2 - 1
    rot = np.dot(rot, np.linalg.qr(np.random.randn(3, 3))[0])
    return rot


# ===== New: PLY-only lightweight reader + Dataset =====
class _SimpleData:
    """模拟 PyG 的 Data：只需要 .pos（Tensor[N,3]）即可兼容后续代码"""
    def __init__(self, pos_np: np.ndarray):
        self.pos = torch.from_numpy(pos_np.astype(np.float32, copy=False))


def _read_ply_points(path: str) -> np.ndarray:
    import struct

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Bad PLY (no end_header): {path}")
            header_lines.append(line.decode("ascii", "ignore").rstrip("\r\n"))
            if line.startswith(b"end_header"):
                break
        header = header_lines

        # --- parse header ---
        fmt = None
        n_vert = None
        in_vertex = False
        vertex_props = []  # list of (type, name) in order
        for ln in header:
            if ln.startswith("format"):
                if "ascii" in ln:
                    fmt = "ascii"
                elif "binary_little_endian" in ln:
                    fmt = "bin_le"
                else:
                    raise ValueError(f"Unsupported PLY format: {ln}")
            elif ln.startswith("element vertex"):
                n_vert = int(ln.split()[-1])
                in_vertex = True
            elif ln.startswith("element "):
                # next element, stop collecting vertex props
                in_vertex = False
            elif in_vertex and ln.startswith("property"):
                toks = ln.split()
                # property <type> <name>
                if len(toks) >= 3:
                    vertex_props.append((toks[1], toks[2]))

        if fmt is None or n_vert is None:
            raise ValueError(f"PLY header parse failed: {path}")

        # 我们只关心前三个属性的数值作为 xyz
        xyz_idx = []
        for i, (_, name) in enumerate(vertex_props):
            if name in ("x", "y", "z"):
                xyz_idx.append(i)
        if len(xyz_idx) < 3:
            # 容忍某些写法：前 3 个属性就是 x y z（即使没按名字）
            xyz_idx = [0, 1, 2]

        if fmt == "ascii":
            pts = []
            for _ in range(n_vert):
                row = f.readline().decode("ascii", "ignore").strip()
                while row == "":
                    row = f.readline().decode("ascii", "ignore").strip()
                vals = row.replace(",", " ").split()
                # 转 float，取 xyz_idx
                floats = []
                for v in vals:
                    try:
                        floats.append(float(v))
                    except ValueError:
                        floats.append(0.0)
                xyz = [floats[i] if i < len(floats) else 0.0 for i in xyz_idx[:3]]
                pts.append(xyz)
            return np.asarray(pts, dtype=np.float32)

        # --- binary little endian ---
        # 为每个 property 计算其字节宽度并形成每顶点步长
        type2size = {
            "char":1, "uchar":1, "int8":1, "uint8":1,
            "short":2, "ushort":2, "int16":2, "uint16":2,
            "int":4, "uint":4, "int32":4, "uint32":4,
            "float":4, "float32":4,
            "double":8, "float64":8
        }
        # 只考虑非 list 的标量 property；list 情况极少见于顶点
        sizes = [type2size.get(t, 4) for (t, _) in vertex_props]
        stride = sum(sizes)
        if stride <= 0:
            raise ValueError(f"Bad vertex stride in {path}")

        # 构造每个属性的 struct 格式（小端）
        type2fmt = {
            1:"b",  # 用有符号占位，uchar 读取后转换为 float 不影响 xyz
            2:"h",
            4:"f",
            8:"d",
        }
        fmts = []
        for (t, _) in vertex_props:
            sz = type2size.get(t, 4)
            if   sz == 1: fmts.append("b" if "char" in t or "int8" in t else "B")
            elif sz == 2: fmts.append("h" if "short" in t and "u" not in t else "H")
            elif sz == 4: fmts.append("f")
            elif sz == 8: fmts.append("d")
            else:         fmts.append("f")
        fmt_struct = "<" + "".join(fmts)
        rec_size = struct.calcsize(fmt_struct)

        # 若 header sizes 求和和 struct 不符，以 struct 为准
        if rec_size != stride:
            stride = rec_size

        data = f.read(n_vert * stride)
        if len(data) < n_vert * stride:
            # 容忍性截断
            n_vert = len(data) // stride
            data = data[: n_vert * stride]

        pts = np.empty((n_vert, 3), dtype=np.float32)
        off = 0
        for i in range(n_vert):
            rec = struct.unpack_from(fmt_struct, data, off)
            off += stride
            # 取前三个 xyz 索引
            x = float(rec[xyz_idx[0]]) if xyz_idx[0] < len(rec) else 0.0
            y = float(rec[xyz_idx[1]]) if xyz_idx[1] < len(rec) else 0.0
            z = float(rec[xyz_idx[2]]) if xyz_idx[2] < len(rec) else 0.0
            pts[i] = (x, y, z)
        return pts



class PLYDirDataset(data.Dataset):
    """
    遍历 exports/modelnet40_{split}_all/<class>/<split>/*.ply
    返回 _SimpleData(pos=torch.Tensor[N,3])，与 PyG 的 Data 读法兼容。
    """
    def __init__(self, root: str, split: str, num_points: int):
        self.root = os.path.abspath(root)
        self.split = split.lower()  # 'train' / 'test'
        self.num_points = int(num_points)

        # <root>/<class>/<split>/*.ply
        classes = sorted([d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d)) and not d.startswith(".")])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        files = []
        for c in classes:
            cdir = os.path.join(self.root, c, self.split)
            if not os.path.isdir(cdir):
                continue
            for fp in sorted(glob.glob(os.path.join(cdir, "*.ply"))):
                files.append((fp, self.class_to_idx[c]))
        if not files:
            raise FileNotFoundError(f"No PLY found under {self.root}/<class>/{self.split}/*.ply")
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp, _ = self.files[idx]
        pc = _read_ply_points(fp).astype(np.float32, copy=False)

        # 兜底：强制成固定点数
        n = pc.shape[0]
        target = self.num_points
        if n == 0:
            pc = np.zeros((target, 3), dtype=np.float32)
        elif n < target:
            reps, rem = divmod(target, n)
            pc = np.concatenate([pc] * reps + ([pc[:rem]] if rem else []), axis=0)
        elif n > target:
            choice = np.random.choice(n, size=target, replace=False)
            pc = pc[choice]

        return _SimpleData(pc)


# ===== Base =====
class ModelNetBase(data.Dataset):
    """A base ModelNet data loader."""

    def __init__(self, data_config, sele_config, **kwargs):
        # 基础配置
        if 'coord_min' in data_config or 'coord_max' in data_config:
            self.coord_minmax = [data_config.get('coord_min', 0), data_config.get('coord_max', 1023)]
        else:
            self.coord_minmax = None
        self.centralize = data_config.get('centralize', True)
        self.voxelize = data_config.get('voxelize', False)
        self.sparse_collate = data_config.get('sparse_collate', False)
        self.augmentation = data_config[sele_config].get('augmentation', False)
        self.split = data_config[sele_config]['split'].lower()
        self.num_points = data_config['num_points']
        self.debug_print = bool(data_config.get('debug_print', False))  # NEW: 调试打印

        # 新增：开关与根目录（默认启用 PLY 导出目录）
        use_ply_exports = data_config.get('use_ply_exports', True)
        ply_root = data_config.get(
            'ply_root',
            os.path.abspath(os.path.join(BASE_DIR, f'../../exports/modelnet40_{self.split}_all'))
        )

        if use_ply_exports:
            self.point_cloud_dataset = PLYDirDataset(root=ply_root, split=self.split, num_points=self.num_points)
        else:
            sampler = SamplePoints(num=self.num_points, remove_faces=True, include_normals=False)
            self.point_cloud_dataset = ModelNet(
                root=dataset_path_default, name='40',
                train=True if self.split == 'train' else False, transform=sampler
            )

    def __len__(self):
        return len(self.point_cloud_dataset)

    def pc_preprocess(self, pc):
        """Perform different types of pre-processings to the ModelNet point clouds."""
        if self.debug_print:
            print("[DEBUG] raw points:", pc.shape)

        if self.centralize:
            centroid = np.mean(pc, axis=0)
            pc = pc - centroid

        if self.augmentation:  # random rotation
            pc = np.dot(pc, gen_rotate())

        if self.coord_minmax is not None:
            # 逐轴缩放，避免某一维 span≈0 造成极端挤压
            pc_min = np.min(pc, axis=0)
            pc_max = np.max(pc, axis=0)
            span = pc_max - pc_min
            span[span == 0] = 1.0
            pc = (pc - pc_min) / span * (self.coord_minmax[1] - self.coord_minmax[0]) + self.coord_minmax[0]

            if self.voxelize:
                # 只有当 voxelize=True 时，才做整数化+去重
                pc = np.unique(np.round(pc).astype('int32'), axis=0)
                if self.sparse_collate:
                    pc = np.hstack((np.zeros((pc.shape[0], 1), dtype='int32'), pc))
                    pc[0][0] = 1
                if self.debug_print:
                    print("[DEBUG] after voxelize:", pc.shape)
                return pc
        else:  # normalize within a unit ball
            m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
            if m > 0:
                pc = pc / m

        if self.debug_print:
            print("[DEBUG] after preprocess:", pc.shape)
        return pc.astype('float32')


# ===== Simple (points) =====
class ModelNetSimple(ModelNetBase):
    """A simple ModelNet data loader where point clouds are directly represented as 3D points."""

    def __init__(self, data_config, sele_config, **kwargs):
        super().__init__(data_config, sele_config)

        # Use_cache specifies the pickle file to be read/written down, "" means no caching mechanism is used
        self.use_cache = data_config.get('use_cache', '')

        # CHANGED: 根据配置选择缓存 dtype，避免“盲目 uint8”
        if self.use_cache != '':
            cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../datasets/', self.use_cache)
            if os.path.exists(cache_file):
                logger.log.info("Loading pre-processed ModelNet40 cache file...")
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = []
                logger.log.info("Sampling point clouds from provided dataset (PLY or OFF)...")

                # 决定缓存 dtype 的规则：
                # - voxelize=True 且 coord_max<=255  -> uint8
                # - voxelize=True 且 coord_max<=65535 -> uint16
                # - 其它情况用 float32，避免精度丢失
                cache_dtype = np.float32
                if self.voxelize and self.coord_minmax is not None:
                    if self.coord_minmax[1] <= 255:
                        cache_dtype = np.uint8
                    elif self.coord_minmax[1] <= 65535:
                        cache_dtype = np.uint16

                for i in range(len(self.point_cloud_dataset)):
                    arr = self.pc_preprocess(self.point_cloud_dataset[i].pos.numpy())
                    self.cache.append(arr.astype(cache_dtype, copy=False))
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            logger.log.info("ModelNet40 data loaded...\n")

    def __getitem__(self, index):
        if self.use_cache:
            arr = self.cache[index]
            # 读取时还原为 float32，后续网络通常期望 float
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            return arr
        else:
            pc = self.pc_preprocess(self.point_cloud_dataset[index].pos.numpy())
            return pc


# ===== Octree =====
class ModelNetOctree(ModelNetBase):
    """ModelNet data loader with uniform sampling and octree partitioning."""

    def __init__(self, data_config, sele_config, **kwargs):
        # CHANGED: 不再强制覆盖 YAML；尊重外部配置
        # data_config['voxelize'] = True
        # data_config['sparse_collate'] = False
        super().__init__(data_config, sele_config)

        self.rw_octree = data_config.get('rw_octree', False)
        if self.rw_octree:
            self.rw_partition_scheme = data_config.get('rw_partition_scheme', 'default')
        self.octree_cache_folder = 'octree_cache'

        # Create an octree formatter to organize octrees into arrays
        self.octree_organizer = OctreeOrganizer(
            data_config['octree_cfg'],
            data_config[sele_config].get('max_num_points', data_config['num_points']),
            kwargs['syntax'].syntax_gt,
            self.rw_octree,
            data_config[sele_config].get('shuffle_blocks', False),
        )

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        while True:
            if self.rw_octree:
                file_name = os.path.join(dataset_path_default, self.octree_cache_folder, self.rw_partition_scheme, str(index)) + '.pkl'
            else:
                file_name = None

            pc = self.pc_preprocess(self.point_cloud_dataset[index].pos.numpy())
            if self.debug_print:
                print("[DEBUG] before octree organize:", pc.shape)

            pc_formatted, _, _, _, all_skip = self.octree_organizer.organize_data(pc, file_name=file_name)
            if all_skip:
                index += 1
                if index >= len(self.point_cloud_dataset):
                    index = 0
            else:
                break

        return pc_formatted
