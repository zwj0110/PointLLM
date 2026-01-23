# Copyright (c) 2010-2022, InterDigital
# All rights reserved.

# See LICENSE under the root folder.

# Codec for the GRASP-Net

import os
import sys
import time
import numpy as np

# Need to put it here due to unknown conflict with MinkowskiEngine
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/nndistance'))

import torch
import MinkowskiEngine as ME
from pccai.codecs.pcc_codec import PccCodecBase
from pccai.models.utils_sparse import slice_sparse_tensor

try:
    import faiss
    found_FAISS = True
except ModuleNotFoundError:
    found_FAISS = False


class GeoResCompressionCodec(PccCodecBase):
    """
    Geometric Residual Analysis and Synthesis for PCC, m58962, Jan 22, the codec itself
    """

    AFFINE_MAGIC = 0x000ACAF0  # sidecar magic

    def __init__(self, codec_config, pccnet, bit_depth, syntax):
        super().__init__(codec_config, pccnet, syntax)
        self.res = 2 ** bit_depth
        self.bit_depth = bit_depth
        pccnet.base_only = codec_config.get('base_only', False)  # whether to use FAISS for NN search
        if pccnet.skip_mode == False:
            # overwrite the option of whether to use FAISS for NN search
            pccnet.res_enc.faiss = codec_config.get('faiss', True) and found_FAISS == True

        # Set the slice parameter
        self.slice = codec_config.get('slice', 0)

        # runtime cache for per-slice affine when reading
        self._affine_cache = {}

    # ------------------------------ helpers ------------------------------

    @staticmethod
    def _write_affine_sidecar(path_base: str, translate: np.ndarray, scale: float):
        """
        Write per-slice affine parameters to a sidecar file, without touching original headers.
        Sidecar layout:
            int32 magic
            float32 translate[3]
            float32 scale[1]
        """
        sidecar = path_base + "._AFFINE.bin"
        with open(sidecar, "wb") as f:
            f.write(np.array([GeoResCompressionCodec.AFFINE_MAGIC], dtype=np.int32).tobytes())
            f.write(np.asarray(translate, dtype=np.float32).tobytes())
            f.write(np.asarray([scale], dtype=np.float32).tobytes())
        return sidecar

    @staticmethod
    def _read_affine_sidecar(path_base: str):
        sidecar = path_base + "._AFFINE.bin"
        if not os.path.exists(sidecar):
            return None
        try:
            with open(sidecar, "rb") as f:
                magic = np.frombuffer(f.read(4), dtype=np.int32)
                if magic.size != 1 or int(magic[0]) != GeoResCompressionCodec.AFFINE_MAGIC:
                    return None
                t = np.frombuffer(f.read(12), dtype=np.float32)  # 3 * 4
                s = np.frombuffer(f.read(4), dtype=np.float32)   # 1 * 4
                if t.size != 3 or s.size != 1:
                    return None
                return {'translate': t.astype(np.float32), 'scale': float(s[0])}
        except Exception:
            return None

    # ------------------------------ core codec ------------------------------

    def compress(self, coords, tag):
        """
        Compress all the transform blocks in a point cloud and write the bitstream to a file.
        """
        # ===== 预处理：范围打印 & 自适应映射到 12-bit 栅格（每个样本独立） =====
        start = time.monotonic()

        arr = np.asarray(coords, dtype=np.float32)
        mn, mx, mean = arr.min(0), arr.max(0), arr.mean(0)
        print(f"[RANGE CHECK] min={mn}, max={mx}, mean={mean}")

        extent = np.maximum(mx - mn, 1e-6)
        extent_max = float(extent.max())
        peak = float(self.res - 1)
        safety = 0.9995

        # 每个样本自己的 affine
        translate_local = (-mn).astype(np.float32)              # 把 min 平到 0
        scale_local = (peak * safety) / extent_max              # 最大轴向占满 12-bit（留安全边）

        # 映射 + 量化到体素网格（Minkowski 需要整数坐标）
        coords_q = (arr + translate_local) * scale_local
        coords_q = np.floor(coords_q + 0.5).astype(np.int32)

        print(f"[DEBUG] slice={self.slice!r} (type={type(self.slice)})")
        pnt_cnt = coords_q.shape[0]
        coords_t = torch.from_numpy(coords_q)  # int32
        feats_t  = torch.ones((len(coords_t), 1), dtype=torch.float32)

        # 构造稀疏张量
        coords_t, feats_t = ME.utils.sparse_collate([coords_t], [feats_t])
        device = next(self.pccnet.parameters()).device
        x_list = ME.SparseTensor(features=feats_t, coordinates=coords_t, tensor_stride=1, device=device)
        x_list = slice_sparse_tensor(x_list, self.slice)
        end = time.monotonic()

        # ===== 统计 =====
        filename_list = []
        stat_dict = {
            'scaled_num_points': 0,
            'all_enc_time': end - start,
            'base_enc_time': 0,
            'bpp_base': 0
        }

        # ===== 按 slice 编码 =====
        for cnt_slice, x in enumerate(x_list):
            start = time.monotonic()
            filename_base, string_set, min_v_set, max_v_set, shape_set, scaled_num_points, base_enc_time = \
                self.pccnet.compress(x, tag + '_' + str(cnt_slice))
            end = time.monotonic()

            stat_dict['scaled_num_points'] += scaled_num_points
            stat_dict['all_enc_time'] += end - start
            stat_dict['base_enc_time'] += base_enc_time
            stat_dict['bpp_base'] += os.stat(filename_base).st_size * 8 / pnt_cnt
            filename_list.append(filename_base)

            # —— 写增强串 & header（保持原格式）——
            if self.pccnet.skip_mode == False:
                filename_enhance = tag + '_' + str(cnt_slice) + '_E' + '.bin'
                filename_header  = tag + '_' + str(cnt_slice) + '_H' + '.bin'

                with open(filename_enhance, 'wb') as fout:
                    for cnt, string in enumerate(string_set):
                        fout.write(string)
                        key = 'bpp_feat'
                        if cnt >= 1:
                            key += '_res' + str(len(string_set) - cnt - 1)
                        if key not in stat_dict:
                            stat_dict[key] = 0
                        stat_dict[key] += len(string) * 8 / pnt_cnt

                with open(filename_header, 'wb') as fout:
                    for cnt in range(len(string_set)):
                        fout.write(np.array(shape_set[cnt], dtype=np.int32).tobytes())
                        fout.write(np.array(len(min_v_set[cnt]), dtype=np.int8).tobytes())
                        fout.write(np.array(min_v_set[cnt], dtype=np.float32).tobytes())
                        fout.write(np.array(max_v_set[cnt], dtype=np.float32).tobytes())
                        if cnt != len(string_set) - 1:
                            fout.write(np.array(len(string_set[cnt]), dtype=np.int32).tobytes())

                filename_list.append(filename_enhance)
                filename_list.append(filename_header)

            # —— 写本 slice 的仿射 sidecar（与 base 文件同名，加后缀）——
            self._write_affine_sidecar(filename_base, translate_local, scale_local)

        # ===== 统计格式化 =====
        stat_dict['scaled_num_points'] = stat_dict['scaled_num_points']
        stat_dict['enc_time'] = round(stat_dict['all_enc_time'] - stat_dict['base_enc_time'], 3)
        stat_dict['all_enc_time'] = round(stat_dict['all_enc_time'], 3)
        stat_dict['base_enc_time'] = round(stat_dict['base_enc_time'], 3)
        for k, v in stat_dict.items():
            if k.startswith('bpp_'):
                stat_dict[k] = round(v, 6)

        return filename_list, stat_dict

    def decompress(self, filename):
        """
        Decompress all the transform blocks of a point cloud from a file.
        """
        stat_dict = {
            'all_dec_time': 0,
            'base_dec_time': 0,
        }

        self._affine_cache.clear()

        for cnt_slice in range(2 ** self.slice):
            # 读取增强串和 header（保持原逻辑）
            if self.pccnet.skip_mode == False:
                with open(filename[cnt_slice * 3 + 1], 'rb') as fin:
                    string_set = [fin.read()]
                shape_set, min_v_set, max_v_set = [], [], []
                with open(filename[cnt_slice * 3 + 2], 'rb') as fin:
                    shape_set.append(np.frombuffer(fin.read(4 * 2), dtype=np.int32))
                    len_min_v = int(np.frombuffer(fin.read(1), dtype=np.int8)[0])
                    # 与 entropy_bottleneck 接口匹配：取标量
                    min_v_set.append(float(np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)[0]))
                    max_v_set.append(float(np.frombuffer(fin.read(4 * len_min_v), dtype=np.float32)[0]))
                    # 不再强行读取“下一个字符串长度”，以避免对原格式的依赖
            else:
                string_set, min_v_set, max_v_set, shape_set = None, None, None, None

            # 查找本 slice 的 affine sidecar
            base_idx = cnt_slice * (1 if self.pccnet.skip_mode else 3)
            base_path = filename[base_idx]
            aff = self._read_affine_sidecar(base_path)
            if aff is not None:
                self._affine_cache[cnt_slice] = aff
            else:
                self._affine_cache[cnt_slice] = None  # fallback to global

            # 解码
            base_dec_time = [0]
            start = time.monotonic()
            if cnt_slice == 0:
                pc_rec = self.postprocess(
                    self.pccnet.decompress(filename[base_idx], string_set, min_v_set, max_v_set, shape_set, base_dec_time),
                    slice_id=cnt_slice
                )
            else:
                pc_rec = torch.vstack((
                    pc_rec,
                    self.postprocess(
                        self.pccnet.decompress(filename[base_idx], string_set, min_v_set, max_v_set, shape_set, base_dec_time),
                        slice_id=cnt_slice
                    )
                ))
            end = time.monotonic()

            stat_dict['all_dec_time'] += end - start
            stat_dict['base_dec_time'] += base_dec_time[0]

        stat_dict['dec_time'] = round(stat_dict['all_dec_time'] - stat_dict['base_dec_time'], 3)
        stat_dict['all_dec_time'] = round(stat_dict['all_dec_time'], 3)
        stat_dict['base_dec_time'] = round(stat_dict['base_dec_time'], 3)
        return pc_rec, stat_dict

    def postprocess(self, pc_rec, slice_id: int = 0):
        """
        Postprocessing after the point cloud is decompressed
        """
        # 先量化到整数并裁剪到 [0, res-1]
        pc_rec = pc_rec.round().long()

        # 统计裁剪（可帮助定位“竖线/截断”的直接原因）
        over = (pc_rec >= self.res).sum(dim=0)
        under = (pc_rec < 0).sum(dim=0)

        pc_rec[pc_rec[:, 0] >= self.res, 0] = self.res - 1
        pc_rec[pc_rec[:, 1] >= self.res, 1] = self.res - 1
        pc_rec[pc_rec[:, 2] >= self.res, 2] = self.res - 1
        pc_rec[pc_rec[:, 0] < 0, 0] = 0
        pc_rec[pc_rec[:, 1] < 0, 1] = 0
        pc_rec[pc_rec[:, 2] < 0, 2] = 0

        if int(over.sum() + under.sum()) > 0:
            print(f"[WARN] Clipped points -> over:{over.tolist()} under:{under.tolist()} / res={self.res}")

        # 去重：把 (x,y,z) 打平成单索引去重，再还原
        lin = pc_rec[:, 0] * (self.res ** 2) + pc_rec[:, 1] * self.res + pc_rec[:, 2]
        lin = torch.unique(lin)
        out0 = torch.floor(lin / (self.res ** 2)).long()
        lin  = lin - out0 * (self.res ** 2)
        out1 = torch.floor(lin / self.res).long()
        lin  = lin - out1 * self.res
        pc_rec = torch.cat([out0.unsqueeze(1), out1.unsqueeze(1), lin.unsqueeze(1)], dim=1)

        # 反归一化：优先使用本 slice 的局部 affine，否则回退全局配置
        aff = self._affine_cache.get(slice_id, None) if hasattr(self, "_affine_cache") else None
        if aff is not None:
            t = torch.tensor(aff['translate'], device=pc_rec.device, dtype=torch.float32)
            s = float(aff['scale'])
            pc_rec = pc_rec.float() / s - t
        else:
            pc_rec = pc_rec.float() / float(self.scale) - torch.tensor(
                self.translate, device=pc_rec.device, dtype=torch.float32
            )

        return pc_rec.long()
