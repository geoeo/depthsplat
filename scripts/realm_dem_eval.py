#!/usr/bin/env python3
"""Compare depth maps with the OpenREALM reference elevation model (DEM) in UTM.

The DEM is the product of an earlier OpenREALM reconstruction of the same flight.
It is a reference, NOT ground truth: it has holes, outliers and systematic
errors of its own (for example over forest and at the mosaic border), so every
number here is a *disagreement* between two reconstructions, not an error of
the model. Read it next to the held-out novel-view scores and the cross-view
depth consistency, which do not depend on the old reconstruction.

Works on scenes built by scripts/realm_gs_scenes.py (cameras.json carries
`anchor_c2w_utm`, so the metric, anchor-relative poses can be put back into
UTM zone 33N). Two checks:

1. Nadir check (no rendering needed): for every frame that has a depth map,
   camera height above the reference DEM (camera z - DEM median around the
   camera) versus the predicted depth at the principal point.
2. Height difference: every depth pixel is unprojected with K and the absolute
   camera pose into UTM, the DEM is sampled at (E, N), and point_z - DEM_z is
   reported (median, mean abs, RMS, fraction within 1 m / 3 m).

Depth maps are camera z-depth in metres: the encoder's prediction at the
context views (images/<scene>/depth/<idx>.npy, --source encoder) or the
rasterised depth at the target views (images/<scene>/depth_rendered/<idx>.npy,
--source rendered, needs test.save_rendered_depth=true).

    python scripts/realm_dem_eval.py --run-dir outputs/<run> \
        --data-root datasets/custom_images_gs --source encoder

The default DEM is custom/realm_2_100m_w_tandem/gnd_1/mosaicing/elevation/gtiff/elevation.tif
(0.15 m/px, EPSG:32633, NaN nodata); only scenes of that flight overlap it.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEM = REPO_ROOT / "custom" / "realm_2_100m_w_tandem" / "gnd_1" / "mosaicing" / "elevation" / "gtiff" / "elevation.tif"
DEPTH_FILE = re.compile(r"^(\d+)\.npy$")
SOURCE_SUBDIR = {"encoder": "depth", "rendered": "depth_rendered"}


class Dem:
    """GeoTIFF DEM with the geotransform taken from the ModelTiepoint/PixelScale tags."""

    def __init__(self, path: Path, min_valid: float):
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            self.z = page.asarray().astype(np.float64)
            tiepoint = page.tags["ModelTiepointTag"].value
            scale = page.tags["ModelPixelScaleTag"].value
            nodata = page.tags.get("GDAL_NODATA")
        i, j, _, e0, n0, _ = tiepoint[:6]
        self.sx, self.sy = float(scale[0]), float(scale[1])
        self.e0 = float(e0) - i * self.sx  # easting of the top-left pixel corner
        self.n0 = float(n0) + j * self.sy  # northing of the top-left pixel corner
        if nodata is not None and str(nodata.value).strip().lower() != "nan":
            self.z[self.z == float(nodata.value)] = np.nan
        self.z[self.z < min_valid] = np.nan  # OpenREALM leaves ~200 m outliers in the tails
        self.rows, self.cols = self.z.shape

    def median_around(self, e: float, n: float, radius_m: float) -> float:
        """Median DEM height within a square of half-width radius_m around UTM (E, N); NaN if nothing valid.

        Single pixels are noisy over trees and at the mosaic border, so the nadir check uses this.
        """
        col = int((e - self.e0) / self.sx)
        row = int((self.n0 - n) / self.sy)
        r = max(int(radius_m / self.sx), 1)
        patch = self.z[max(row - r, 0): row + r + 1, max(col - r, 0): col + r + 1]
        return float(np.nanmedian(patch)) if patch.size and np.isfinite(patch).any() else float("nan")

    def sample(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Nearest-pixel DEM height for UTM (E, N); NaN outside or on nodata."""
        col = np.floor((e - self.e0) / self.sx).astype(np.int64)
        row = np.floor((self.n0 - n) / self.sy).astype(np.int64)
        inside = (col >= 0) & (col < self.cols) & (row >= 0) & (row < self.rows)
        out = np.full(e.shape, np.nan)
        out[inside] = self.z[row[inside], col[inside]]
        return out

    @property
    def valid_fraction(self) -> float:
        return float(np.isfinite(self.z).mean())


def unproject(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, step: int) -> np.ndarray:
    """Unproject a z-depth map (subsampled by `step`) to world points [N, 3]."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float64)
    pix = np.stack([u + 0.5, v + 0.5, np.ones_like(u, dtype=np.float64)], axis=-1).reshape(-1, 3)
    rays = pix @ np.linalg.inv(K).T  # camera-space directions with z = 1
    pts_cam = rays * z.reshape(-1, 1)
    pts_world = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    return pts_world


def principal_point_depth(depth: np.ndarray, K: np.ndarray, radius: int = 2) -> float:
    cx, cy = int(round(K[0, 2])), int(round(K[1, 2]))
    patch = depth[max(cy - radius, 0): cy + radius + 1, max(cx - radius, 0): cx + radius + 1]
    return float(np.median(patch))


def evaluate_scene(scene_dir: Path, depth_dir: Path, dem: Dem, step: int, radius_m: float) -> dict:
    cameras = json.loads((scene_dir / "cameras.json").read_text())
    if "anchor_c2w_utm" not in cameras:
        raise ValueError(f"{scene_dir}: cameras.json has no anchor_c2w_utm (build scenes with realm_gs_scenes.py)")
    anchor = np.array(cameras["anchor_c2w_utm"], dtype=np.float64)
    frames = {f["index"]: f for f in cameras["frames"]}

    result = {"scene": cameras["scene"], "frames": []}
    all_err = []
    for npy in sorted(depth_dir.glob("*.npy")):
        m = DEPTH_FILE.match(npy.name)
        if not m:
            continue
        index = int(m.group(1))
        if index not in frames:
            continue
        frame = frames[index]
        K = np.array(frame["intrinsics"], dtype=np.float64)
        c2w = anchor @ np.array(frame["extrinsics"], dtype=np.float64)
        depth = np.load(npy)
        cam_e, cam_n, cam_z = c2w[:3, 3]

        dem_under_camera = dem.median_around(cam_e, cam_n, radius_m)
        agl = cam_z - dem_under_camera if np.isfinite(dem_under_camera) else np.nan
        pp_depth = principal_point_depth(depth, K)

        pts = unproject(depth, K, c2w, step)
        dem_z = dem.sample(pts[:, 0], pts[:, 1])
        valid = np.isfinite(dem_z) & np.isfinite(pts[:, 2])
        err = pts[valid, 2] - dem_z[valid]
        all_err.append(err)
        entry = {
            "index": index,
            "frame_id": frame.get("frame_id"),
            "role": frame.get("role"),
            "camera_utm": [cam_e, cam_n, cam_z],
            "dem_under_camera": dem_under_camera,
            "agl_from_dem": agl,
            "principal_point_depth": pp_depth,
            "depth_over_agl": pp_depth / agl if np.isfinite(agl) and agl > 0 else np.nan,
            "n_points": int(valid.size),
            "dem_coverage": float(valid.mean()) if valid.size else 0.0,
        }
        if err.size:
            entry.update({
                "height_diff_median": float(np.median(err)),
                "height_diff_mean_abs": float(np.mean(np.abs(err))),
                "height_diff_rms": float(np.sqrt(np.mean(err ** 2))),
                "within_1m": float(np.mean(np.abs(err) < 1.0)),
                "within_3m": float(np.mean(np.abs(err) < 3.0)),
            })
        result["frames"].append(entry)

    err = np.concatenate(all_err) if all_err else np.array([])
    result["overall"] = {
        "n_points": int(err.size),
        "height_diff_median": float(np.median(err)) if err.size else None,
        "height_diff_mean_abs": float(np.mean(np.abs(err))) if err.size else None,
        "height_diff_rms": float(np.sqrt(np.mean(err ** 2))) if err.size else None,
        "within_1m": float(np.mean(np.abs(err) < 1.0)) if err.size else None,
        "within_3m": float(np.mean(np.abs(err) < 3.0)) if err.size else None,
    }
    return result


def print_report(results: list[dict], dem: Dem, source: str) -> None:
    print(f"DEM valid fraction {dem.valid_fraction:.3f}, pixel {dem.sx} m, origin E {dem.e0:.1f} N {dem.n0:.1f}")
    print(f"depth source: {source}\n")
    print("reference DEM = earlier OpenREALM reconstruction, not ground truth; values are disagreements\n")
    print("scene | frame | role | camera height above reference DEM (m) | depth at principal point (m) | depth/height | DEM coverage | median height diff (m) | mean |diff| | RMS | <1 m | <3 m")
    for r in results:
        for f in r["frames"]:
            cov = f["dem_coverage"]
            stats = (f"{f['height_diff_median']:+.2f} | {f['height_diff_mean_abs']:.2f} | {f['height_diff_rms']:.2f} | "
                     f"{f['within_1m']:.2f} | {f['within_3m']:.2f}") if "height_diff_median" in f else "- | - | - | - | -"
            print(f"{r['scene']} | {f['index']} | {f['role']} | {f['agl_from_dem']:.1f} | {f['principal_point_depth']:.1f} | "
                  f"{f['depth_over_agl']:.2f} | {cov:.2f} | {stats}")
        o = r["overall"]
        if o["n_points"]:
            print(f"{r['scene']} | all | | | | | | {o['height_diff_median']:+.2f} | {o['height_diff_mean_abs']:.2f} | "
                  f"{o['height_diff_rms']:.2f} | {o['within_1m']:.2f} | {o['within_3m']:.2f}")
        else:
            print(f"{r['scene']} | all | no reference DEM overlap")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Test-run output directory (contains images/<scene>/)")
    parser.add_argument("--data-root", type=Path, required=True, help="custom_images root whose test/<scene>/cameras.json has anchor_c2w_utm")
    parser.add_argument("--scene", default=None, help="Only this scene (default: every scene in the run)")
    parser.add_argument("--source", choices=sorted(SOURCE_SUBDIR), default="encoder")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--min-valid", type=float, default=300.0, help="DEM heights below this (m) are treated as nodata")
    parser.add_argument("--step", type=int, default=4, help="Pixel subsampling for the unprojection")
    parser.add_argument("--nadir-radius", type=float, default=10.0,
                        help="Half-width (m) of the DEM patch whose median gives the height under the camera")
    parser.add_argument("--output", type=Path, default=None, help="JSON report (default <run-dir>/dem_eval_<source>.json)")
    args = parser.parse_args()

    dem = Dem(args.dem, args.min_valid)
    results = []
    for scene_images in sorted((args.run_dir / "images").iterdir()):
        if args.scene and scene_images.name != args.scene:
            continue
        depth_dir = scene_images / SOURCE_SUBDIR[args.source]
        scene_dir = args.data_root / "test" / scene_images.name
        if not depth_dir.is_dir() or not (scene_dir / "cameras.json").is_file():
            continue
        results.append(evaluate_scene(scene_dir, depth_dir, dem, args.step, args.nadir_radius))
    if not results:
        raise SystemExit("no scene with depth maps and a cameras.json carrying anchor_c2w_utm")

    print_report(results, dem, args.source)
    output = args.output or args.run_dir / f"dem_eval_{args.source}.json"
    output.write_text(json.dumps({"dem": str(args.dem), "source": args.source,
                                  "note": "reference DEM from an earlier OpenREALM reconstruction, not ground truth; "
                                          "height_diff = point_z - dem_z is a disagreement between two reconstructions",
                                  "scenes": results}, indent=2))
    print(f"\nreport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
