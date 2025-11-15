#!/usr/bin/env python3
"""
From a SOURCE ZIP, collect only *.ply files and write them into a new ZIP
with the layout:

    ModelNet40/<class>/test/<filename>.ply

- <class> is inferred from the path inside the source ZIP:
  * If path looks like ".../<class>/test/<file>.ply" -> use that <class>
  * Else if path looks like "<top>/<class>/<file>.ply" -> use the 2nd segment
  * Else fall back to the parent folder name.

Usage:
  python zip_collect_ply.py --src_zip /path/to/source.zip --out_zip ModelNet40_ply_test.zip
  # Only include *_dec.ply:
  # python zip_collect_ply.py --src_zip source.zip --out_zip out.zip --only_dec
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED

def infer_class(parts: list[str]) -> str:
    """
    Infer class name from the path parts inside the source zip.
    Examples (-> class):
      ['output_dense_lossless_modelnet40', 'airplane', 'airplane_0001_dec.ply'] -> airplane
      ['something', 'chair', 'test', 'chair_0001.ply'] -> chair
      ['chair', 'chair_0001.ply'] -> chair
    """
    if len(parts) >= 3 and parts[-2].lower() in ("test", "train"):
        return parts[-3]
    if len(parts) >= 2:
        # common case: <top>/<class>/<file>
        return parts[1]
    if len(parts) >= 1:
        # fallback to parent (single-level)
        return parts[0]
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_zip", required=True, type=Path, help="Source ZIP containing .ply files.")
    ap.add_argument("--out_zip", required=True, type=Path, help="Output ZIP to write.")
    ap.add_argument("--dst_top", default="ModelNet40", help="Top-level folder name inside output ZIP.")
    ap.add_argument("--only_dec", action="store_true", help="Keep only files ending with '_dec.ply'.")
    ap.add_argument("--compress", action="store_true", help="Use ZIP_DEFLATED (smaller but slower).")
    args = ap.parse_args()

    src_zip: Path = args.src_zip
    out_zip: Path = args.out_zip
    dst_top: str = args.dst_top

    if not src_zip.exists():
        raise SystemExit(f"[Error] Source zip not found: {src_zip}")

    compression = ZIP_DEFLATED if args.compress else ZIP_STORED

    count = 0
    with ZipFile(src_zip, "r") as zin, ZipFile(out_zip, "w", compression=compression) as zout:
        for info in zin.infolist():
            name = info.filename
            if not name.lower().endswith(".ply"):
                continue
            if args.only_dec and not name.lower().endswith("_dec.ply"):
                continue

            parts = [p for p in name.split("/") if p not in ("", ".")]
            cls = infer_class(parts)
            filename = parts[-1]

            arcname = f"{dst_top}/{cls}/test/{filename}"

            # Stream-copy to output without extracting to disk
            with zin.open(info, "r") as fsrc, zout.open(arcname, "w") as fdst:
                shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
            count += 1

    print(f"[OK] Wrote {out_zip}  (files included: {count})")

if __name__ == "__main__":
    main()
