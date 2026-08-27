#!/usr/bin/env python3
"""Compare candidate depth .npy predictions to a reference (up-to-scale).

Finds the least-squares scale factor that best maps candidate depth onto the
reference depth, then reports scale-invariant error metrics.
"""

import argparse
from pathlib import Path

import numpy as np


def load_stack(depth_dir: Path) -> np.ndarray:
    files = sorted(depth_dir.glob("[0-9]*.npy"))
    if not files:
        raise FileNotFoundError(f"No npy files found in {depth_dir}")
    return np.stack([np.load(f) for f in files], axis=0), files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("reference_dir", type=Path)
    args = parser.parse_args()

    cand, cand_files = load_stack(args.candidate_dir)
    ref, ref_files = load_stack(args.reference_dir)

    if cand.shape != ref.shape:
        # resize candidate to reference resolution if needed (nearest via numpy indexing)
        from scipy.ndimage import zoom
        zoom_factors = (1,) + tuple(r / c for r, c in zip(ref.shape[1:], cand.shape[1:]))
        cand = zoom(cand, zoom_factors, order=1)

    c = cand.reshape(-1).astype(np.float64)
    r = ref.reshape(-1).astype(np.float64)

    # least-squares scale: minimize ||s*c - r||^2 -> s = (c . r) / (c . c)
    scale = float((c * r).sum() / (c * c).sum())
    scaled = c * scale

    rmse = float(np.sqrt(np.mean((scaled - r) ** 2)))
    rel_rmse = rmse / float(r.mean())
    mae = float(np.mean(np.abs(scaled - r)))
    rel_mae = mae / float(r.mean())
    corr = float(np.corrcoef(c, r)[0, 1])
    # fraction of pixels within 25% of reference after scaling (a1-like metric)
    ratio = np.maximum(scaled / r, r / scaled)
    a1 = float(np.mean(ratio < 1.25))

    print(f"candidate: {args.candidate_dir}")
    print(f"reference: {args.reference_dir}")
    print(f"n_frames={cand.shape[0]} shape={cand.shape[1:]}")
    print(f"optimal_scale={scale:.6g}")
    print(f"corr={corr:.4f}  rel_rmse={rel_rmse:.4f}  rel_mae={rel_mae:.4f}  a1(<1.25x)={a1:.4f}")
    print(f"candidate raw range: min={c.min():.4g} max={c.max():.4g} mean={c.mean():.4g}")
    print(f"reference raw range: min={r.min():.4g} max={r.max():.4g} mean={r.mean():.4g}")


if __name__ == "__main__":
    main()
