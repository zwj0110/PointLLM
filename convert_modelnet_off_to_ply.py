#!/usr/bin/env python3
"""
convert_modelnet_off_to_ply.py  (robust OFF reader)

Changes in this version:
- Adds --backend {auto,open3d,trimesh} (default: auto).
- If Open3D fails (e.g., "OFF352 322 0" header), we transparently fall back to trimesh.
- Handles meshes that need triangulation (via trimesh).

Install:
  pip install trimesh plyfile
  # (optional) pip install open3d

Examples:
  python convert_modelnet_off_to_ply.py --modelnet_root /path/to/ModelNet40 --out_root /tmp/pcd --points 20000 --normalize unitcube
  python convert_modelnet_off_to_ply.py --backend trimesh --modelnet_root /path/to/ModelNet40 --out_root /tmp/pcd --points 20000
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

def _import_open3d():
    import open3d as o3d  # may raise ImportError
    return o3d

def _import_trimesh():
    import trimesh  # may raise ImportError
    return trimesh

def _try_imports(prefer):
    prefer = prefer.lower()
    if prefer == "open3d":
        try:
            return ("open3d", _import_open3d())
        except Exception as e:
            raise RuntimeError("Requested backend 'open3d' but import failed. Try --backend auto or --backend trimesh.\n" + str(e))
    if prefer == "trimesh":
        try:
            return ("trimesh", _import_trimesh())
        except Exception as e:
            raise RuntimeError("Requested backend 'trimesh' but import failed. Try `pip install trimesh`.\n" + str(e))
    # auto
    try:
        return ("open3d", _import_open3d())
    except Exception:
        # silently fall back to trimesh
        try:
            return ("trimesh", _import_trimesh())
        except Exception as e:
            raise RuntimeError("Need either 'open3d' or 'trimesh' installed. `pip install trimesh` OR `pip install open3d`") from e

def _fix_off_header_if_needed(off_path):
    """
    Some ModelNet40 OFF files start with e.g. 'OFF352 322 0' on the first line.
    This function creates a temp file with a normalized header if needed and returns its path.
    If no fix is needed, returns the original path.
    """
    with open(off_path, 'r') as f:
        first = f.readline().strip()
        if first == "OFF":
            return off_path
        if first.startswith("OFF") and first != "OFF":
            # e.g., 'OFF352 322 0' -> we need to split into two lines
            rest = f.read()
            counts = first[3:].strip()
            tmp = tempfile.NamedTemporaryFile(suffix=".off", delete=False, mode="w")
            tmp.write("OFF\n")
            if counts:
                tmp.write(counts + "\n")
            tmp.write(rest)
            tmp.flush()
            tmp.close()
            return tmp.name
        # Otherwise leave as is
        return off_path

def _sample_points_open3d(off_path, num_points):
    o3d = _import_open3d()
    fixed = _fix_off_header_if_needed(off_path)
    mesh = o3d.io.read_triangle_mesh(str(fixed))
    if not mesh.has_triangles():
        raise ValueError("open3d: mesh has no triangles")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    import numpy as np
    return o3d.numpy.asarray(pcd.points)

def _sample_points_trimesh(off_path, num_points):
    tm = _import_trimesh()
    mesh = tm.load(str(off_path), force='mesh')
    if not isinstance(mesh, tm.Trimesh):
        # e.g., a Scene -> combine to a single Trimesh if possible
        if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
            mesh = tm.util.concatenate(tuple(mesh.geometry.values()))
        else:
            raise ValueError("trimesh: loaded object is not a mesh")
    # ensure triangulated
    if not mesh.is_watertight or not mesh.is_volume:
        mesh = mesh.as_open3d if hasattr(mesh, 'as_open3d') else mesh
    if not mesh.is_watertight and hasattr(mesh, 'triangles'):
        pass
    # sample
    import numpy as np
    pts, _ = tm.sample.sample_surface(mesh, num_points)
    return pts.astype("float32")

def sample_points_from_off(off_path, num_points, backend):
    name, _ = backend
    # Try preferred backend first; if it fails, fall back to the other if available.
    tried = []
    if name == "open3d":
        tried.append("open3d")
        try:
            return _sample_points_open3d(off_path, num_points)
        except Exception as e:
            # fallback to trimesh
            try:
                return _sample_points_trimesh(off_path, num_points)
            except Exception as e2:
                raise RuntimeError(f"Both open3d and trimesh sampling failed for {off_path}.\nopen3d error: {e}\ntrimesh error: {e2}")
    else:
        tried.append("trimesh")
        try:
            return _sample_points_trimesh(off_path, num_points)
        except Exception as e:
            # fallback to open3d
            try:
                return _sample_points_open3d(off_path, num_points)
            except Exception as e2:
                raise RuntimeError(f"Both trimesh and open3d sampling failed for {off_path}.\ntrimesh error: {e}\nopen3d error: {e2}")

def normalize_points(pts, mode="unitcube"):
    import numpy as np
    pts = pts.astype("float32")
    if mode == "none":
        return pts
    mn = pts.min(axis=0, keepdims=True)
    mx = pts.max(axis=0, keepdims=True)
    size = (mx - mn).max()
    if size <= 1e-12:
        return pts - mn
    pts = (pts - mn) / size  # [0,1]
    return pts

def quantize_points(pts01, resolution):
    import numpy as np
    scale = float(resolution - 1)
    q = (pts01 * scale).round().clip(0, scale).astype("int32")
    return q

def save_ply_float(ply_path, pts):
    import numpy as np
    from plyfile import PlyData, PlyElement
    arr = np.empty(pts.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    arr["x"], arr["y"], arr["z"] = pts[:,0], pts[:,1], pts[:,2]
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(ply_path)

def save_ply_int(ply_path, qpts):
    import numpy as np
    from plyfile import PlyData, PlyElement
    arr = np.empty(qpts.shape[0], dtype=[("x", "i4"), ("y", "i4"), ("z", "i4")])
    arr["x"], arr["y"], arr["z"] = qpts[:,0], qpts[:,1], qpts[:,2]
    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(ply_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelnet_root", type=str, required=True, help="Root containing class folders or OFF files")
    parser.add_argument("--out_root", type=str, required=True, help="Destination root for PLY files")
    parser.add_argument("--points", type=int, default=100000, help="Points per mesh to sample")
    parser.add_argument("--normalize", type=str, default="unitcube", choices=["none", "unitcube"], help="Normalization mode")
    parser.add_argument("--resolution", type=int, default=None, help="If set, quantize to this resolution")
    parser.add_argument("--save_ints", action="store_true", help="Write integer PLY if quantized")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto","open3d","trimesh"], help="Sampling backend preference")
    args = parser.parse_args()

    backend = _try_imports(args.backend)
    modelnet_root = Path(args.modelnet_root)
    out_root = Path(args.out_root)
    off_files = list(modelnet_root.rglob("*.off"))
    if not off_files:
        print(f"No .off files found under {modelnet_root}", file=sys.stderr)
        sys.exit(1)

    for off in off_files:
        rel = off.relative_to(modelnet_root).with_suffix(".ply")
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pts = sample_points_from_off(off, args.points, backend)
        pts = normalize_points(pts, args.normalize)
        if args.resolution is not None:
            qpts = quantize_points(pts, args.resolution)
            if args.save_ints:
                save_ply_int(str(out_path), qpts)
            else:
                save_ply_float(str(out_path), qpts.astype("float32"))
        else:
            save_ply_float(str(out_path), pts.astype("float32"))
        print("Wrote", out_path)

if __name__ == "__main__":
    main()
