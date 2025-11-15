#!/usr/bin/env python3
"""
Make a ZIP of the first N '.off' files from each class's 'test/' folder in a ModelNet40 tree,
preserving the original directory structure inside the ZIP.

Usage:
  python modelnet40_test_subset_zip.py /path/to/ModelNet40 out.zip --per-class 10

Notes:
- "First N" is determined by natural sort of filenames (airplane_2.off < airplane_10.off).
- If a class has fewer than N files, all available files are included.
- Only files under <class>/test/ are considered.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def natural_key(s: str):
    # Split into digit/non-digit chunks to sort "airplane_2.off" before "airplane_10.off"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def make_subset_zip(root: Path, out_zip: Path, per_class: int = 10, subset: str = "test") -> dict:
    root = root.resolve()
    out_zip = out_zip.resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[Error] Input root does not exist or is not a directory: {root}")

    included_counts = {}
    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
        for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            test_dir = cls_dir / subset
            if not test_dir.exists():
                continue

            # Collect .off (case-insensitive)
            files = [p for p in test_dir.iterdir()
                     if p.is_file() and p.suffix.lower() == ".off"]

            # Natural sort, then take first N
            files.sort(key=lambda p: natural_key(p.name))
            selected = files[:per_class]

            # Add to zip with arcname relative to the root
            for f in selected:
                arcname = f.relative_to(root).as_posix()
                zf.write(f, arcname)

            if selected:
                included_counts[cls_dir.name] = len(selected)

    return {
        "root": str(root),
        "out_zip": str(out_zip),
        "per_class": per_class,
        "subset": subset,
        "classes_included": len(included_counts),
        "file_counts": included_counts,
    }

def main():
    parser = argparse.ArgumentParser(description="Zip first N .off files from each class's test/ folder.")
    parser.add_argument("root", type=Path, help="Path to ModelNet40 root directory")
    parser.add_argument("out_zip", type=Path, help="Path to output ZIP file (e.g., test10.zip)")
    parser.add_argument("--per-class", type=int, default=10, help="How many .off files to keep per class (default: 10)")
    parser.add_argument("--subset", type=str, default="test", help="Which subset folder to pull from (default: test)")
    args = parser.parse_args()

    info = make_subset_zip(args.root, args.out_zip, per_class=args.per_class, subset=args.subset)
    print("[Done] Wrote:", info["out_zip"])
    print(" Root:", info["root"])
    print(" Subset:", info["subset"])
    print(" Per class:", info["per_class"])
    print(" Classes included:", info["classes_included"])
    for cls, cnt in sorted(info["file_counts"].items()):
        print(f"  - {cls}: {cnt} files")

if __name__ == "__main__":
    main()
