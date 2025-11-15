#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, argparse
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError as e:
    raise SystemExit("请先安装依赖: pip install plyfile numpy") from e

def find_xyz_element(ply: "PlyData"):
    # 优先常见名字；否则自动遍历
    prefer = ("vertex","point","points","range_grid","cloud","data")
    elems = {e.name: e for e in ply.elements}
    for name in prefer:
        e = elems.get(name)
        if e is not None and {"x","y","z"}.issubset(e.data.dtype.names or ()):
            return e
    for e in ply.elements:
        if {"x","y","z"}.issubset(e.data.dtype.names or ()):
            return e
    return None

def make_xyz_only_element(src_el: "PlyElement"):
    names = src_el.data.dtype.names or ()
    if not {"x","y","z"}.issubset(names):
        raise ValueError("该元素不含 x,y,z")
    # 构造只含 x,y,z 的新 dtype，保持原来每列的数据类型
    new_dtype = [("x", src_el.data.dtype.fields["x"][0]),
                 ("y", src_el.data.dtype.fields["y"][0]),
                 ("z", src_el.data.dtype.fields["z"][0])]
    arr = np.empty(src_el.data.shape[0], dtype=np.dtype(new_dtype))
    for k in ("x","y","z"):
        arr[k] = src_el.data[k]
    # 名字保持原元素名（避免破坏依赖）
    return PlyElement.describe(arr, src_el.name)

def strip_batch_keep_xyz_bytes(in_path: str) -> bytes:
    ply = PlyData.read(in_path)
    target = find_xyz_element(ply)
    if target is None:
        raise ValueError("找不到包含 x,y,z 的元素")

    # 重建元素：只留 x,y,z（不管 batch 在不在、在第几列）
    xyz_el = make_xyz_only_element(target)

    new_elements = []
    for el in ply.elements:
        new_elements.append(xyz_el if el is target else el)

    out = PlyData(new_elements,
                  text=ply.text,               # 保持 ASCII / binary
                  byte_order=ply.byte_order,   # 保持端序
                  comments=list(ply.comments),
                  obj_info=list(ply.obj_info))
    bio = io.BytesIO(); out.write(bio)
    return bio.getvalue()

def iter_ply_files(root):
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".ply"):
                full = os.path.join(r, fn)
                rel = os.path.relpath(full, root)
                yield full, rel

def main():
    ap = argparse.ArgumentParser(description="删除 batch，仅保留 x,y,z")
    ap.add_argument("input_dir")
    ap.add_argument("--inplace", action="store_true", help="覆盖原文件（谨慎）")
    ap.add_argument("--out-dir", default=None, help="输出到目录")
    ap.add_argument("--zip", dest="zip_path", default=None, help="输出到 zip 文件")
    args = ap.parse_args()

    if sum(map(bool,[args.inplace, args.out_dir, args.zip_path])) != 1:
        raise SystemExit("三选一：--inplace | --out-dir DIR | --zip OUT.zip")

    ok = skip = 0
    if args.inplace:
        for full, _ in iter_ply_files(args.input_dir):
            try:
                data = strip_batch_keep_xyz_bytes(full)
                with open(full, "wb") as f: f.write(data)
                ok += 1
            except Exception as e:
                print(f"[WARN] Skip {full}: {e}"); skip += 1
        print(f"完成：OK={ok}, Skip={skip}")
        return

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for full, rel in iter_ply_files(args.input_dir):
            out_path = os.path.join(args.out_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                data = strip_batch_keep_xyz_bytes(full)
                with open(out_path, "wb") as f: f.write(data)
                ok += 1
            except Exception as e:
                print(f"[WARN] Skip {rel}: {e}"); skip += 1
        print(f"输出目录：{args.out_dir} | OK={ok}, Skip={skip}")
        return

    # 写 zip
    with ZipFile(args.zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for full, rel in iter_ply_files(args.input_dir):
            try:
                data = strip_batch_keep_xyz_bytes(full)
                zf.writestr(rel, data)
                ok += 1
            except Exception as e:
                print(f"[WARN] Skip {rel}: {e}"); skip += 1
    print(f"输出 ZIP：{args.zip_path} | OK={ok}, Skip={skip}")

if __name__ == "__main__":
    main()
