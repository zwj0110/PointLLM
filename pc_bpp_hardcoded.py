
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最简单用法：直接改 PATHS 列表里的文件/目录路径，然后运行本脚本即可。
支持 .ply / .off / .npy。输出 num_points、bit_total（比特）、bpp。
"""

from pathlib import Path
import os

# ←←← 只需要改这里：把你的文件/目录路径填进来（可混合）
PATHS = [
    "/Users/zhengwenjie/projects/PointLLM/data/ModelNet40-sparse-dense-lossless/airplane/test/airplane_0627.ply"
    # "/absolute/or/relative/path/to/your_file_or_dir"
]

def file_bits(p: Path) -> int:
    return p.stat().st_size * 8

def parse_ply_num_vertices(p: Path) -> int:
    n = None
    with open(p, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("latin-1", errors="ignore").strip()
            if s.lower().startswith("element vertex"):
                parts = s.split()
                if len(parts) >= 3 and parts[2].lstrip("+-").isdigit():
                    n = int(parts[2])
            if s.lower().startswith("end_header"):
                break
    if n is None:
        raise ValueError(f"PLY header missing 'element vertex N': {p}")
    return n

def is_int_token(tok: str) -> bool:
    t = tok.strip()
    if t.startswith(("+","-")):
        t = t[1:]
    return t.isdigit()

def parse_off_num_vertices(p: Path) -> int:
    # 宽松解析：找到第一行包含至少3个整数的行，把第一个数当作 n_verts
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"Empty OFF: {p}")
    # 跳过首行 'OFF'/'COFF' 等
    idx = 0
    if lines[0].upper().endswith("OFF"):
        idx = 1
    # 从 idx 开始找第一行 3 个整数
    for j in range(idx, len(lines)):
        parts = lines[j].split()
        ints = [q for q in parts if is_int_token(q)]
        if len(ints) >= 3:
            return int(ints[0])
    raise ValueError(f"Could not find counts line with 3 integers in OFF: {p}")

def parse_npy_num_points(p: Path) -> int:
    import numpy as np
    arr = np.load(p, mmap_mode="r")
    if arr.ndim < 2:
        raise ValueError(f"NPY expected 2D array (N,C), got {arr.shape}")
    return int(arr.shape[0])

def metrics_for_file(p: Path):
    ext = p.suffix.lower()
    bits = file_bits(p)
    if ext == ".ply":
        n = parse_ply_num_vertices(p)
    elif ext == ".off":
        n = parse_off_num_vertices(p)
    elif ext == ".npy":
        n = parse_npy_num_points(p)
    else:
        return {"path": str(p), "error": f"unsupported ext {ext}"}
    bpp = bits / n if n > 0 else float("nan")
    return {"path": str(p), "num_points": n, "bit_total": bits, "bpp": bpp}

def iter_targets(pathlike):
    exts = {".ply", ".off", ".npy"}
    p = Path(pathlike)
    if p.is_file():
        yield p
    elif p.is_dir():
        for ext in exts:
            yield from p.rglob(f"*{ext}")
    else:
        print(f"[warn] not found: {p}")

def main():
    rows = []
    for a in PATHS:
        for p in iter_targets(a):
            try:
                rows.append(metrics_for_file(p))
            except Exception as e:
                rows.append({"path": str(p), "error": str(e)})
    # 打印结果
    if not rows:
        print("No files found. Edit PATHS in this script.")
        return
    colw = {
        "path": max(4, max(len(r["path"]) for r in rows)),
        "num_points": 11,
        "bit_total": 12,
        "bpp": 12,
    }
    header = f'{"path".ljust(colw["path"])}  {"num_points".rjust(colw["num_points"])}  {"bit_total".rjust(colw["bit_total"])}  {"bpp".rjust(colw["bpp"])}'
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f'{r["path"].ljust(colw["path"])}  {"-":>{colw["num_points"]}}  {"-":>{colw["bit_total"]}}  {"-":>{colw["bpp"]}}  # {r["error"]}')
        else:
            print(f'{r["path"].ljust(colw["path"])}  {str(r["num_points"]).rjust(colw["num_points"])}  {str(r["bit_total"]).rjust(colw["bit_total"])}  {r["bpp"]:.2f}'.rjust(colw["bpp"]))
    # 同时导出 CSV（在脚本同目录）
    out_csv = Path(__file__).with_suffix(".csv")
    try:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["path", "num_points", "bit_total", "bpp"])
            w.writeheader()
            for r in rows:
                if "error" in r:
                    continue
                w.writerow({k: r[k] for k in ["path", "num_points", "bit_total", "bpp"]})
        print(f"\nSaved CSV: {out_csv}")
    except Exception as e:
        print(f"[warn] CSV write failed: {e}")

if __name__ == "__main__":
    main()
