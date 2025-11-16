#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将压缩过的 PLY 文件（文件名含类名）整理为:
ModelNet40/{class}/test/{class}_{id}.ply

支持输入为 zip 文件或目录。
文件名示例：grasp_airplane_0627_rec.ply → ModelNet40/airplane/test/airplane_0627.ply
"""

import argparse
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional

PATTERN_PREFIX = "grasp_"
PATTERN_SUFFIX = "_rec.ply"

def parse_class_and_newname(filename: str) -> Tuple[str, str]:
    """
    从文件名解析 class 与新的文件名: {class}_{id}.ply
    规则：
      - 去掉前缀 grasp_（如有）
      - 去掉后缀 _rec.ply（如有），保留 .ply
      - 最后一个下划线段如果全是数字，视为编号
    """
    base = os.path.basename(filename)
    name = base

    if name.startswith(PATTERN_PREFIX):
        name = name[len(PATTERN_PREFIX):]

    if name.endswith(PATTERN_SUFFIX):
        core = name[:-len(PATTERN_SUFFIX)]
        ext = ".ply"
    else:
        # 通用兜底：保留 .ply
        ext = ".ply" if name.lower().endswith(".ply") else ""
        core = name[:-4] if name.lower().endswith(".ply") else name

    parts = core.split("_")
    if parts and re.fullmatch(r"\d+", parts[-1] or ""):
        cls = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
        fid = parts[-1]
        new_base = f"{cls}_{fid}{ext}"
    else:
        cls = core
        new_base = f"{core}{ext}"

    return cls, new_base

def ensure_unique_path(dst_path: Path) -> Path:
    """
    若目标已存在，自动追加 _1, _2, ...
    """
    if not dst_path.exists():
        return dst_path
    stem = dst_path.stem
    suffix = dst_path.suffix
    parent = dst_path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def iter_ply_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".ply":
            yield p

def extract_zip(src_zip: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src_zip, "r") as z:
        z.extractall(extract_dir)
    # 选出包含 .ply 的最浅目录作为源根
    candidate = None
    for root, _, files in os.walk(extract_dir):
        if any(f.lower().endswith(".ply") for f in files):
            p = Path(root)
            if candidate is None or len(p.parts) < len(candidate.parts):
                candidate = p
    return candidate or extract_dir

def main():
    ap = argparse.ArgumentParser(
        description="将包含类名的 PLY 文件整理成 ModelNet40/{class}/test 结构"
    )
    ap.add_argument("src", help="源路径：zip 文件或目录")
    ap.add_argument("-o", "--out", default="ModelNet40", help="输出根目录（默认：ModelNet40）")
    ap.add_argument("--dry-run", action="store_true", help="只预览不写入")
    ap.add_argument("--clean-out", action="store_true", help="若输出目录已存在则先删除")
    args = ap.parse_args()

    src_path = Path(args.src).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()

    if args.clean_out and out_root.exists():
        print(f"[info] 清理已存在的输出目录: {out_root}")
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    # 处理源
    temp_dir: Optional[Path] = None
    if src_path.is_file() and src_path.suffix.lower() == ".zip":
        temp_dir = out_root.parent / "__tmp_extract__"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print(f"[info] 正在解压 zip: {src_path}")
        src_root = extract_zip(src_path, temp_dir)
    elif src_path.is_dir():
        src_root = src_path
    else:
        print(f"[error] 无法识别的输入：{src_path}")
        sys.exit(1)

    counts = defaultdict(int)
    total = 0
    moved = 0

    print(f"[info] 扫描源目录: {src_root}")
    for f in iter_ply_files(src_root):
        total += 1
        cls, new_name = parse_class_and_newname(f.name)
        dst_dir = out_root / cls / "test"
        dst_path = dst_dir / new_name

        if not args.dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = ensure_unique_path(dst_path)
            shutil.copy2(f, dst_path)

        counts[cls] += 1
        moved += 1

        if moved % 50 == 0:
            print(f"[info] 已处理 {moved} / 约{total}…")

    # 清理临时目录
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\n====== 汇总 ======")
    print(f"总计扫描文件: {total}")
    print(f"成功输出文件: {moved}")
    print(f"输出根目录  : {out_root}\n")
    print("各类计数：")
    for cls in sorted(counts.keys()):
        print(f"  {cls:20s} {counts[cls]}")

    print("\n完成 ✅")

if __name__ == "__main__":
    main()
