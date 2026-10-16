#!/usr/bin/env python3
"""Build Gaussian-splatting evaluation windows from a realm dataset.

Each window of consecutive keyframes becomes one custom_images scene. Poses are
expressed in metres relative to the window's first context camera, which becomes
the identity matrix; the anchor pose (UTM) is kept in cameras.json so results can
be mapped back. Every scene gets a disjoint context/target split that is written
to an evaluation-index JSON for the `realm` view sampler.

    python scripts/realm_gs_scenes.py --dataset realm_2_with_depth --offset 70 \
        --window 7 --stride 7 --num-windows 4 --context-step 2

There is deliberately no --scale and no near/far option: units are metres, and
near/far come from config/dataset/custom_images.yaml for the whole scene.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from convert_realm_to_custom_images import load_intrinsics, load_trajectory  # noqa: E402

CUSTOM_ROOT = REPO_ROOT / "custom"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "custom_images_gs"
DEFAULT_INDEX_PATH = REPO_ROOT / "assets" / "evaluation_index_realm_gs.json"
REFERENCE_DEPTH_DIR_NAMES = ("depth", "dense")
UTM_EPSG = 32633  # WGS 84 / UTM zone 33N, as in the OpenREALM GeoTIFFs


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def find_reference_depth(depth_dir: Path | None, frame_id: str) -> str | None:
    if depth_dir is None or not depth_dir.is_dir():
        return None
    exact = depth_dir / f"{frame_id}.npy"
    if exact.is_file():
        return str(exact)
    matches = sorted(depth_dir.glob(f"*{frame_id}.npy"))
    return str(matches[0]) if matches else None


def split_window(size: int, context_step: int | None, context_first: int | None) -> tuple[list[int], list[int]]:
    """Return (context, target) window-local indices, disjoint."""
    local = list(range(size))
    if context_first is not None:
        context = local[:context_first]
    else:
        context = local[::context_step]
    target = [i for i in local if i not in context]
    return context, target


def build_window(
    dataset: str,
    source_indices: list[int],
    context_local: list[int],
    target_local: list[int],
    data_root: Path,
    intrinsics: np.ndarray,
    poses: dict[str, np.ndarray],
    image_files: list[Path],
    depth_dir: Path | None,
    max_step: float,
    index_targets: str,
) -> tuple[str | None, dict | None, list[str]]:
    """Write one window as a scene. Returns (scene_name, index_entry, warnings).

    Returns (None, None, warnings) when the window is rejected.
    """
    warnings: list[str] = []
    scene_name = f"{dataset}_w{source_indices[0]:04d}-{source_indices[-1]:04d}"

    frame_ids = [image_files[i].stem for i in source_indices]
    missing = [fid for fid in frame_ids if fid not in poses]
    if missing:
        warnings.append(f"{scene_name}: rejected, no pose for frame(s) {missing}")
        return None, None, warnings

    # Reject windows that span a trajectory break (pose outliers in realm_1 / realm_2).
    positions = np.stack([poses[fid][:3, 3] for fid in frame_ids])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if steps.size and steps.max() > max_step:
        k = int(steps.argmax())
        warnings.append(
            f"{scene_name}: rejected, {steps.max():.1f} m jump between source frames "
            f"{source_indices[k]} and {source_indices[k + 1]} (limit {max_step} m)"
        )
        return None, None, warnings

    anchor_local = context_local[0]
    anchor_id = frame_ids[anchor_local]
    T0 = poses[anchor_id]
    T0_inv = se3_inverse(T0)

    scene_dir = data_root / "test" / scene_name
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True)

    frames = []
    for local, (src_index, fid) in enumerate(zip(source_indices, frame_ids)):
        c2w = T0_inv @ poses[fid]
        out_name = f"{local:03d}.png"
        shutil.copy2(image_files[src_index], scene_dir / out_name)
        frame = {
            "image": out_name,
            "intrinsics": intrinsics.tolist(),
            "extrinsics": c2w.tolist(),
            "index": local,
            "source_index": src_index,
            "frame_id": fid,
            "role": "context" if local in context_local else "target",
        }
        reference = find_reference_depth(depth_dir, fid)
        if reference is not None:
            frame["depth_reference"] = reference
        frames.append(frame)

    # Loose tolerance: UTM translations are ~1e6, the subtraction leaves mm-level residuals.
    anchor_rel = np.array(frames[anchor_local]["extrinsics"])
    if not np.allclose(anchor_rel, np.eye(4), atol=1e-4):
        raise AssertionError(f"{scene_name}: anchor pose is not identity after the relative transform")

    context_positions = np.stack([np.array(frames[i]["extrinsics"])[:3, 3] for i in context_local])
    baseline = float(np.linalg.norm(context_positions[-1] - context_positions[0]))

    cameras = {
        "scene": scene_name,
        "dataset": dataset,
        "units": "m",
        "utm_epsg": UTM_EPSG,
        "anchor_frame_id": anchor_id,
        "anchor_source_index": source_indices[anchor_local],
        "anchor_c2w_utm": T0.tolist(),
        "source_indices": source_indices,
        "context_indices": context_local,
        "target_indices": target_local,
        "context_baseline_m": baseline,
        "frames": frames,
    }
    with (scene_dir / "cameras.json").open("w") as f:
        json.dump(cameras, f, indent=2)

    index_entry = {
        "context": context_local,
        "target": list(range(len(source_indices))) if index_targets == "all" else target_local,
    }
    return scene_name, index_entry, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Folder name under custom/")
    parser.add_argument("--offset", type=int, default=0, help="Source index of the first window's first frame")
    parser.add_argument("--window", type=int, default=7, help="Consecutive keyframes per window")
    parser.add_argument("--stride", type=int, default=None, help="Source-index step between windows (default: --window)")
    parser.add_argument("--num-windows", type=int, default=1)
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--context-step", type=int, default=2,
                       help="Interleaved split: every k-th window frame is context, the rest are targets (default 2)")
    split.add_argument("--context-first", type=int, default=None,
                       help="Extrapolation split: the first n window frames are context, the rest are targets")
    parser.add_argument("--index-targets", choices=["heldout", "all"], default="heldout",
                        help="Index target list: held-out frames only (default) or every window frame "
                             "(input-view reprojection check; inflates PSNR)")
    parser.add_argument("--max-step", type=float, default=40.0,
                        help="Reject windows with a larger jump (m) between consecutive keyframes")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--keep-existing", action="store_true",
                        help="Keep other scenes under <data-root>/test and merge into an existing index "
                             "(default: start from an empty test root and index)")
    args = parser.parse_args()

    if args.window < 2:
        parser.error("--window must be at least 2")
    if args.context_first is not None and not 1 <= args.context_first < args.window:
        parser.error("--context-first must be in [1, window)")
    if args.context_first is None and args.context_step < 1:
        parser.error("--context-step must be at least 1")
    stride = args.stride or args.window

    source_dir = CUSTOM_ROOT / args.dataset
    if not source_dir.is_dir():
        parser.error(f"no such dataset folder: {source_dir}")
    intrinsics = load_intrinsics(source_dir / "intrinsics.txt")
    poses = load_trajectory(source_dir / "kf_traj.txt")
    image_files = sorted((source_dir / "imgs").glob("*.png"))
    depth_dir = next((source_dir / n for n in REFERENCE_DEPTH_DIR_NAMES if (source_dir / n).is_dir()), None)
    print(f"dataset {args.dataset}: {len(image_files)} images, {len(poses)} poses, "
          f"reference depth: {depth_dir if depth_dir else 'none'}")

    context_local, target_local = split_window(args.window, args.context_step, args.context_first)
    if not target_local and args.index_targets == "heldout":
        parser.error("the split leaves no held-out target; use a larger --context-step, a smaller "
                     "--context-first, or --index-targets all")

    test_root = args.data_root / "test"
    index: dict[str, dict] = {}
    if args.keep_existing:
        if args.index_path.is_file():
            index = json.loads(args.index_path.read_text())
    elif test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    rows = []
    all_warnings: list[str] = []
    for k in range(args.num_windows):
        start = args.offset + k * stride
        source_indices = list(range(start, start + args.window))
        if source_indices[-1] >= len(image_files):
            all_warnings.append(f"window {k}: source indices {start}-{source_indices[-1]} exceed "
                                f"{len(image_files)} images, stopped")
            break
        scene_name, entry, warnings = build_window(
            args.dataset, source_indices, context_local, target_local, args.data_root,
            intrinsics, poses, image_files, depth_dir, args.max_step, args.index_targets,
        )
        all_warnings.extend(warnings)
        if scene_name is None:
            continue
        index[scene_name] = entry
        cameras = json.loads((test_root / scene_name / "cameras.json").read_text())
        rows.append((scene_name, source_indices, entry["context"], entry["target"], cameras["context_baseline_m"]))

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    with args.index_path.open("w") as f:
        json.dump(index, f, indent=2)

    print("\nscene | source indices | context | target | context baseline (m)")
    for scene_name, src, ctx, tgt, baseline in rows:
        print(f"{scene_name} | {src[0]}-{src[-1]} | {ctx} | {tgt} | {baseline:.1f}")
    for warning in all_warnings:
        print(f"WARNING: {warning}")
    print(f"\n{len(rows)} scene(s) under {test_root}\nindex: {args.index_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
