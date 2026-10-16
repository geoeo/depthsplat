#!/usr/bin/env python3
"""Multi-view consistency of depth maps, using only the known poses.

For every ordered pair of views (i, j) of a scene, the z-depth of view i is
unprojected, transformed into view j with the relative poses from
cameras.json, projected with K, and the predicted z in j is compared with
view j's own depth map sampled at the projected pixel. A depth map that is
geometrically consistent across views gives a high inlier fraction and
correlation; this needs no ground truth, so it works on every realm scene.

    python scripts/depth_consistency.py --run-dir outputs/<run> \
        --data-root datasets/custom_images_gs --source encoder

Depth sources: the encoder prediction at the context views
(images/<scene>/depth/<idx>.npy) or the rasterised depth at the target views
(images/<scene>/depth_rendered/<idx>.npy). Reference depth copies
(<idx>_reference.npy) can be scored with --source reference; they are only
up to scale, so read the correlation, not the inlier fraction.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

SOURCE = {
    "encoder": ("depth", re.compile(r"^(\d+)\.npy$")),
    "rendered": ("depth_rendered", re.compile(r"^(\d+)\.npy$")),
    "reference": ("depth", re.compile(r"^(\d+)_reference\.npy$")),
}


def load_views(scene_images: Path, scene_dir: Path, source: str) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    cameras = json.loads((scene_dir / "cameras.json").read_text())
    frames = {f.get("index", i): f for i, f in enumerate(cameras["frames"])}
    subdir, pattern = SOURCE[source]
    depths, poses, K = {}, {}, None
    for npy in sorted((scene_images / subdir).glob("*.npy")):
        m = pattern.match(npy.name)
        if not m or int(m.group(1)) not in frames:
            continue
        index = int(m.group(1))
        depths[index] = np.load(npy).astype(np.float64)
        poses[index] = np.array(frames[index]["extrinsics"], dtype=np.float64)
        K = np.array(frames[index]["intrinsics"], dtype=np.float64)
    return depths, poses, K


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = u - 0.5
    y = v - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
    out = np.full(u.shape, np.nan)
    xa, ya, x0, y0, x1, y1 = x[inside], y[inside], x0[inside], y0[inside], x1[inside], y1[inside]
    wx, wy = xa - x0, ya - y0
    out[inside] = (
        img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
        + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy
    )
    return out


def pair_consistency(depth_i, c2w_i, depth_j, c2w_j, K, step, tol) -> dict:
    h, w = depth_i.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth_i[::step, ::step].reshape(-1)
    pix = np.stack([u + 0.5, v + 0.5, np.ones_like(u, dtype=np.float64)], axis=-1).reshape(-1, 3)
    pts_cam_i = (pix @ np.linalg.inv(K).T) * z[:, None]
    pts_world = pts_cam_i @ c2w_i[:3, :3].T + c2w_i[:3, 3]
    w2c_j = np.linalg.inv(c2w_j)
    pts_cam_j = pts_world @ w2c_j[:3, :3].T + w2c_j[:3, 3]
    z_j = pts_cam_j[:, 2]
    front = z_j > 1e-6
    proj = (pts_cam_j[front] @ K.T) / z_j[front, None]
    sampled = bilinear(depth_j, proj[:, 0], proj[:, 1])
    ok = np.isfinite(sampled)
    z_pred, z_obs = z_j[front][ok], sampled[ok]
    if z_pred.size < 10:
        return {"n": int(z_pred.size), "overlap": float(ok.mean()) if ok.size else 0.0}
    rel = np.abs(z_pred - z_obs) / np.maximum(z_obs, 1e-6)
    return {
        "n": int(z_pred.size),
        "overlap": float(ok.sum() / z.size),
        "median_rel_err": float(np.median(rel)),
        "inlier_frac": float(np.mean(rel < tol)),
        "corr": float(np.corrcoef(z_pred, z_obs)[0, 1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--source", choices=sorted(SOURCE), default="encoder")
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--tol", type=float, default=0.05, help="Relative depth error counted as inlier")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {"source": args.source, "tol": args.tol, "scenes": {}}
    print(f"depth source: {args.source}, inlier tolerance {args.tol:.0%}\n")
    print("scene | pair i->j | overlap | median rel err | inlier frac | corr")
    for scene_images in sorted((args.run_dir / "images").iterdir()):
        if args.scene and scene_images.name != args.scene:
            continue
        scene_dir = args.data_root / "test" / scene_images.name
        if not (scene_dir / "cameras.json").is_file() or not (scene_images / SOURCE[args.source][0]).is_dir():
            continue
        depths, poses, K = load_views(scene_images, scene_dir, args.source)
        if len(depths) < 2:
            continue
        pairs = {}
        for i in depths:
            for j in depths:
                if i == j:
                    continue
                r = pair_consistency(depths[i], poses[i], depths[j], poses[j], K, args.step, args.tol)
                pairs[f"{i}->{j}"] = r
                if "corr" in r:
                    print(f"{scene_images.name} | {i}->{j} | {r['overlap']:.2f} | {r['median_rel_err']:.3f} | {r['inlier_frac']:.2f} | {r['corr']:.3f}")
                else:
                    print(f"{scene_images.name} | {i}->{j} | {r['overlap']:.2f} | (too little overlap)")
        scored = [r for r in pairs.values() if "corr" in r]
        summary = {
            "n_pairs": len(scored),
            "median_rel_err": float(np.mean([r["median_rel_err"] for r in scored])) if scored else None,
            "inlier_frac": float(np.mean([r["inlier_frac"] for r in scored])) if scored else None,
            "corr": float(np.mean([r["corr"] for r in scored])) if scored else None,
        }
        if scored:
            print(f"{scene_images.name} | mean over {len(scored)} pairs | | {summary['median_rel_err']:.3f} | "
                  f"{summary['inlier_frac']:.2f} | {summary['corr']:.3f}")
        report["scenes"][scene_images.name] = {"pairs": pairs, "summary": summary}

    if not report["scenes"]:
        raise SystemExit("no scene with at least two depth maps")
    output = args.output or args.run_dir / f"depth_consistency_{args.source}.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
