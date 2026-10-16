#!/usr/bin/env python3
"""Re-express an exported Gaussian PLY in UTM (EPSG:32633) with a local origin.

`save_gaussian_ply` writes Gaussians in the frame of the first context camera
(positions rotated by that camera's inverse rotation, no translation). For
scenes built by scripts/realm_gs_scenes.py the first context camera is the
anchor (identity), so the PLY is already in the anchor-relative metric frame;
this script applies the anchor pose from cameras.json to move it to UTM, then
subtracts a local origin so float32 coordinates keep millimetre precision, and
records the origin and EPSG code in a sidecar JSON.

    python scripts/export_gaussians_utm.py \
        --ply outputs/<run>/gaussians/<scene>.ply \
        --scene-dir datasets/custom_images_gs/test/<scene>

Only the DC colour band is in the PLY (export_ply drops the higher SH bands),
so no spherical-harmonics rotation is needed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True, help="Scene folder with cameras.json (anchor_c2w_utm)")
    parser.add_argument("--output", type=Path, default=None, help="Default: <ply stem>_utm.ply next to the input")
    parser.add_argument("--origin", choices=["anchor", "none"], default="anchor",
                        help="Local origin subtracted from UTM positions: the anchor camera position rounded "
                             "down to whole metres (default) or none (raw UTM, float32 precision loss)")
    args = parser.parse_args()

    cameras = json.loads((args.scene_dir / "cameras.json").read_text())
    if "anchor_c2w_utm" not in cameras:
        raise SystemExit("cameras.json has no anchor_c2w_utm; build the scene with scripts/realm_gs_scenes.py")
    anchor = np.array(cameras["anchor_c2w_utm"], dtype=np.float64)
    frames = cameras["frames"]
    first_context = cameras.get("context_indices", [0])[0]
    first_c2w = np.array(frames[first_context]["extrinsics"], dtype=np.float64)

    ply = PlyData.read(args.ply)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    quat_wxyz = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)

    # Undo export_ply's rotation into the first context camera, then apply the anchor pose.
    R_first = first_c2w[:3, :3]
    R_anchor, t_anchor = anchor[:3, :3], anchor[:3, 3]
    xyz_world = xyz @ R_first.T
    xyz_utm = xyz_world @ R_anchor.T + t_anchor

    rot = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()  # scipy wants x, y, z, w
    rot_utm = R_anchor @ R_first @ rot
    q = R.from_matrix(rot_utm).as_quat()  # x, y, z, w
    quat_out = np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=1)

    origin = np.floor(t_anchor) if args.origin == "anchor" else np.zeros(3)
    xyz_local = xyz_utm - origin

    out = v.copy()
    out["x"], out["y"], out["z"] = (xyz_local[:, i].astype(np.float32) for i in range(3))
    for i in range(4):
        out[f"rot_{i}"] = quat_out[:, i].astype(np.float32)

    output = args.output or args.ply.with_name(args.ply.stem + "_utm.ply")
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(out, "vertex")]).write(output)
    sidecar = {
        "epsg": cameras.get("utm_epsg", 32633),
        "units": "m",
        "origin_utm": origin.tolist(),
        "note": "position_utm = position + origin_utm; rotations are world (UTM) quaternions (w, x, y, z); "
                "scales are log-scales in metres as written by export_ply",
        "source_ply": str(args.ply),
        "scene": cameras["scene"],
        "anchor_frame_id": cameras.get("anchor_frame_id"),
        "n_gaussians": int(len(out)),
        "bbox_utm_min": (xyz_utm.min(axis=0)).tolist(),
        "bbox_utm_max": (xyz_utm.max(axis=0)).tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {output} ({len(out)} gaussians), origin {origin.tolist()}, "
          f"bbox UTM min {np.round(xyz_utm.min(axis=0), 1).tolist()} max {np.round(xyz_utm.max(axis=0), 1).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
