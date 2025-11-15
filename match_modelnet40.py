#!/usr/bin/env python3
"""
Identify which ModelNet40 *test* object your target point cloud matches,
by comparing geometry (rotation/translation/scale-invariant) against candidates.

Usage (two-stage, multithreaded):
  python -u match_modelnet40.py --target /path/to/your_target.ply \
      --root /path/to/ModelNet40/bathtub/test \
      --ext off ply --topk 5 --quant 0.001 --points 22000 \
      --workers 8 --progress-every 10 --show-file-steps --shape-topN 20 --two-stage

Single-pass (also multithreaded, but slower than two-stage on large sets):
  python -u match_modelnet40.py ... --single-pass
"""

import argparse, os, sys, hashlib, numpy as np, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import open3d as o3d
except Exception as e:
    print("This script requires open3d. Install: pip install open3d scikit-learn", file=sys.stderr, flush=True)
    sys.exit(2)

def read_points(path: Path, target_points: int = 22000) -> np.ndarray:
    """Read PLY points or sample points from OFF mesh."""
    if path.suffix.lower() == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        P = np.asarray(pcd.points, dtype=np.float64)
        return P
    if path.suffix.lower() == ".off":
        mesh = o3d.io.read_triangle_mesh(str(path))
        n = max(20000, target_points)
        try:
            pcd = mesh.sample_points_poisson_disk(number_of_points=n, init_factor=5)
        except Exception:
            pcd = mesh.sample_points_uniformly(number_of_points=n)
        return np.asarray(pcd.points, dtype=np.float64)
    raise ValueError(f"Unsupported extension: {path.suffix}")

def pca_align(points: np.ndarray) -> np.ndarray:
    """Center, scale to unit sphere, PCA-align, fix signs and right-handedness."""
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(axis=0, keepdims=True)
    Pc = P - c
    r = np.linalg.norm(Pc, axis=1).max() or 1.0
    Pu = Pc / r
    cov = np.cov(Pu.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # Stable sign rule
    for i in range(3):
        col = eigvecs[:, i]
        j = np.argmax(np.abs(col))
        if col[j] < 0:
            eigvecs[:, i] = -col
    # Right-handed basis
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] = -eigvecs[:, 2]
    return Pu @ eigvecs

def eigval_ratios(Pn: np.ndarray) -> np.ndarray:
    """Eigenvalue ratios of normalized points (quick shape descriptor)."""
    cov = np.cov(Pn.T)
    w, _ = np.linalg.eigh(cov)
    w = np.sort(w)[::-1]
    m = w.max() or 1.0
    return w / m

def geom_hash(points: np.ndarray, quant: float = 1e-3, sample_cap: int = 50000) -> str:
    """Order-invariant hash on PCA-aligned, quantized points."""
    Q = points
    if len(Q) > sample_cap:
        idx = np.random.default_rng(42).choice(len(Q), size=sample_cap, replace=False)
        Q = Q[idx]
    Qq = np.round(Q / quant).astype(np.int64)
    order = np.lexsort((Qq[:, 2], Qq[:, 1], Qq[:, 0]))
    Qqs = Qq[order].T.copy()
    return hashlib.sha256(Qqs.tobytes()).hexdigest()

def chamfer(A: np.ndarray, B: np.ndarray, sample_cap: int = 8192) -> float:
    """Symmetric Chamfer distance using scikit-learn NN (CPU)."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    kA = min(len(A), sample_cap)
    kB = min(len(B), sample_cap)
    A_ = A[rng.choice(len(A), kA, replace=False)] if len(A) > kA else A
    B_ = B[rng.choice(len(B), kB, replace=False)] if len(B) > kB else B
    nnA = NearestNeighbors(n_neighbors=1).fit(B_)
    nnB = NearestNeighbors(n_neighbors=1).fit(A_)
    da, _ = nnA.kneighbors(A_, return_distance=True)
    db, _ = nnB.kneighbors(B_, return_distance=True)
    return float(da.mean() + db.mean())

def _fmt_sec(s: float) -> str:
    if s < 1:
        return f"{int(s * 1000)} ms"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path, help="Folder with ModelNet40 class test objects (e.g., ModelNet40/bathtub/test)")
    ap.add_argument("--ext", nargs="+", default=["off", "ply"], help="Candidate extensions to consider")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--quant", type=float, default=1e-3)
    ap.add_argument("--points", type=int, default=22000, help="Target #points for OFF sampling")

    # Progress & performance options
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))),
                    help="Number of threads for parallel processing (default=min(8, CPU cores))")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N completed files")
    ap.add_argument("--show-file-steps", dest="show_file_steps", action="store_true",
                    help="Print per-file steps (READ / PCA+HASH / CHAMFER) when available")
    ap.add_argument("--chamfer-cap", type=int, default=8192, help="Max points per set when computing Chamfer")

    # Modes
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--two-stage", dest="two_stage", action="store_true",
                   help="Fast path: stage-1 hash+shape for all, then stage-2 Chamfer on top-N")
    g.add_argument("--single-pass", dest="single_pass", action="store_true",
                   help="Single pass: compute Chamfer for all (slower on large sets)")
    ap.add_argument("--shape-topN", type=int, default=20, help="When --two-stage, only Chamfer the top-N by shape")

    args = ap.parse_args()
    if not args.two_stage and not args.single_pass:
        args.two_stage = True  # default

    # Load target & its descriptors
    T = read_points(args.target, target_points=args.points)
    Tn = pca_align(T)
    thash = geom_hash(Tn, quant=args.quant)
    Tr = eigval_ratios(Tn)

    print(f"Target: {args.target}", flush=True)
    print(f" - vertices: {len(T)}", flush=True)
    print(f" - geometry_hash(quant={args.quant}): {thash}", flush=True)

    # Gather candidates
    cand_files = []
    for ext in args.ext:
        cand_files.extend(sorted(args.root.glob(f"*.{ext}")))
    total = len(cand_files)
    if not total:
        print("No candidate files found under:", args.root, file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"Found {total} candidates under {args.root}", flush=True)
    print(f"Using {args.workers} worker threads", flush=True)

    # ---------- Two-stage (recommended) ----------
    if args.two_stage:
        t_start = time.perf_counter()
        hash_matches = []
        quick_rows = []  # (shape_score, Path)

        def stage1_task(fp: Path):
            # Return (fp, h, shape_score); heavy math in NumPy/Open3D releases GIL ⇒ good for threads
            P = read_points(fp, target_points=args.points)
            Pn = pca_align(P)
            h  = geom_hash(Pn, quant=args.quant)
            R  = eigval_ratios(Pn)
            shape_score = float(np.sum(np.abs(Tr - R)))
            return fp, h, shape_score

        # Submit all stage-1 tasks
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fp in cand_files:
                futures.append(ex.submit(stage1_task, fp))

            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    fp, h, sscore = fut.result()
                    if h == thash:
                        hash_matches.append(fp)
                        print(f"[{done}/{total}] ✅ HASH MATCH: {fp.name}", flush=True)
                    else:
                        quick_rows.append((sscore, fp))
                    if args.show_file_steps:
                        print(f"[{done}/{total}] STAGE-1  {fp.name}  shape={sscore:.6f}", flush=True)
                except Exception as e:
                    print(f"[warn] stage-1: {e}", file=sys.stderr, flush=True)

                if (done % args.progress_every == 0) or (done == total):
                    elapsed = time.perf_counter() - t_start
                    rate = done / max(elapsed, 1e-9)
                    rem = (total - done) / max(rate, 1e-9)
                    print(f"Progress (stage-1): {done}/{total} "
                          f"({done/total:.1%})  elapsed {_fmt_sec(elapsed)}  avg {rate:.2f}/s  ETA {_fmt_sec(rem)}",
                          flush=True)

        if hash_matches:
            print("\n=== Exact geometry-hash matches (rotation/translation/scale-invariant) ===", flush=True)
            for fp in hash_matches:
                print(fp.name, flush=True)
            return

        # Stage-2: Chamfer only top-N by shape
        quick_rows.sort(key=lambda x: x[0])
        subset = [fp for _, fp in quick_rows[:max(1, args.shape_topN)]]
        print(f"\nStage-2: computing Chamfer on top-{len(subset)} shape candidates...", flush=True)

        def stage2_task(fp: Path):
            P = read_points(fp, target_points=args.points)
            Pn = pca_align(P)
            d = chamfer(Tn, Pn, sample_cap=args.chamfer_cap)
            return fp, d

        rows = []
        t2_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures2 = [ex.submit(stage2_task, fp) for fp in subset]
            done2 = 0
            for fut in as_completed(futures2):
                done2 += 1
                try:
                    fp, d = fut.result()
                    rows.append((d, fp))
                    if args.show_file_steps:
                        print(f"[{done2}/{len(subset)}] STAGE-2 CHAMFER {fp.name} -> {d:.6f}", flush=True)
                except Exception as e:
                    print(f"[warn] stage-2: {e}", file=sys.stderr, flush=True)

                if (done2 % max(1, args.progress_every)) == 0 or (done2 == len(subset)):
                    elapsed = time.perf_counter() - t2_start
                    rate = done2 / max(elapsed, 1e-9)
                    rem = (len(subset) - done2) / max(rate, 1e-9)
                    print(f"Progress (stage-2): {done2}/{len(subset)} "
                          f"({done2/len(subset):.1%})  elapsed {_fmt_sec(elapsed)}  avg {rate:.2f}/s  ETA {_fmt_sec(rem)}",
                          flush=True)

        rows.sort(key=lambda x: x[0])
        print("\n=== Top-k nearest by normalized Chamfer distance ===", flush=True)
        for i, (d, fp) in enumerate(rows[:args.topk], 1):
            print(f"{i:2d}. {fp.name:<40}  chamfer={d:.6f}", flush=True)
        return

    # ---------- Single-pass (also threaded, computes Chamfer for all) ----------
    else:
        t_start = time.perf_counter()
        hash_matches = []
        dists = []

        def single_task(fp: Path):
            P = read_points(fp, target_points=args.points)
            Pn = pca_align(P)
            h  = geom_hash(Pn, quant=args.quant)
            if h == thash:
                return ("match", fp, 0.0)
            d = chamfer(Tn, Pn, sample_cap=args.chamfer_cap)
            return ("dist", fp, d)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(single_task, fp) for fp in cand_files]
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    kind, fp, val = fut.result()
                    if kind == "match":
                        hash_matches.append(fp)
                        print(f"[{done}/{total}] ✅ HASH MATCH: {fp.name}", flush=True)
                    else:
                        dists.append((val, fp))
                        if args.show_file_steps:
                            print(f"[{done}/{total}] CHAMFER {fp.name} -> {val:.6f}", flush=True)
                except Exception as e:
                    print(f"[warn] single-pass: {e}", file=sys.stderr, flush=True)

                if (done % args.progress_every == 0) or (done == total):
                    elapsed = time.perf_counter() - t_start
                    rate = done / max(elapsed, 1e-9)
                    rem = (total - done) / max(rate, 1e-9)
                    print(f"Progress: {done}/{total} "
                          f"({done/total:.1%})  elapsed {_fmt_sec(elapsed)}  avg {rate:.2f}/s  ETA {_fmt_sec(rem)}",
                          flush=True)

        if hash_matches:
            print("\n=== Exact geometry-hash matches (rotation/translation/scale-invariant) ===", flush=True)
            for fp in hash_matches:
                print(fp.name, flush=True)
            return

        dists.sort(key=lambda x: x[0])
        print("\n=== Top-k nearest by normalized Chamfer distance ===", flush=True)
        for i, (d, fp) in enumerate(dists[:args.topk], 1):
            print(f"{i:2d}. {fp.name:<40}  chamfer={d:.6f}", flush=True)

if __name__ == "__main__":
    main()
