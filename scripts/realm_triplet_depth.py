#!/usr/bin/env python3
"""Triplet-wise depth inference on realm_1 / realm_2.

Takes a flat list of image indices (length 3*N, relative to --offset) or a --count of
consecutive indices, splits it into triplets, converts each triplet into its own
custom_images scene with extrinsics expressed relative to the triplet's first frame,
then runs DepthSplat depth inference in this same process.
"""

import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# --- xformers gating -------------------------------------------------------
# XFORMERS_DISABLED is read at *import* time: once by
# src/model/encoder/unimatch/ldm_unet/cross_attention.py (pulled in by the
# src.depth_inference import below) and once by the vendored DINOv2 layers that
# torch.hub loads during model construction. Setting it any later silently does
# nothing and yields a false pass, so --xformers is pre-parsed here rather than
# in main(), before the first import that reads it.
#
# The fallback is torch's scaled_dot_product_attention, which torch.export
# lowers to aten::scaled_dot_product_attention; xformers' memory_efficient_
# attention is an opaque custom op that cannot be exported.
XFORMERS_MODES = ("auto", "on", "off")

XFORMERS_GATED_MODULES = {
    "local cross_attention": "src.model.encoder.unimatch.ldm_unet.cross_attention",
    "dinov2 attention": "dinov2.layers.attention",
    "dinov2 block": "dinov2.layers.block",
}


def resolve_xformers_mode(argv: list[str]) -> str:
    """Pre-parse --xformers out of argv, before argparse exists."""
    mode = "auto"
    for i, token in enumerate(argv):
        if token == "--xformers" and i + 1 < len(argv):
            mode = argv[i + 1]
        elif token.startswith("--xformers="):
            mode = token.split("=", 1)[1]
    if mode not in XFORMERS_MODES:
        raise SystemExit(f"--xformers must be one of {XFORMERS_MODES}, got {mode!r}")
    return mode


def apply_xformers_mode(mode: str) -> None:
    """Set the process environment. Must run before any gated module is imported."""
    if mode == "off":
        os.environ["XFORMERS_DISABLED"] = "1"
    elif mode == "on":
        os.environ.pop("XFORMERS_DISABLED", None)
    # "auto": inherit whatever the caller exported, including nothing.


def xformers_state() -> dict[str, bool | None]:
    """Actual XFORMERS_AVAILABLE per gated module; None if not imported yet.

    Reads sys.modules rather than importing, so calling this early does not
    pull in DINOv2 before torch.hub has built the model.
    """
    return {
        label: getattr(sys.modules[name], "XFORMERS_AVAILABLE", None) if name in sys.modules else None
        for label, name in XFORMERS_GATED_MODULES.items()
    }


def print_xformers_state(when: str) -> None:
    env = os.environ.get("XFORMERS_DISABLED")
    print(f"  xformers [{when}]: mode={XFORMERS_MODE} XFORMERS_DISABLED={env!r}")
    for label, available in xformers_state().items():
        shown = "not imported" if available is None else f"XFORMERS_AVAILABLE={available}"
        print(f"    {label:22s} {shown}")


XFORMERS_MODE = resolve_xformers_mode(sys.argv[1:])
apply_xformers_mode(XFORMERS_MODE)

from convert_realm_to_custom_images import load_intrinsics, load_trajectory
from src.depth_inference import compose_config, depth_inference_overrides, run_depth_inference

CUSTOM_ROOT = REPO_ROOT / "custom"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "custom_images_triplets"
DATASET_CFG_PATH = REPO_ROOT / "config" / "dataset" / "custom_images.yaml"
GT_DEPTH_DIR_NAMES = ("depth", "dense")


def parse_indices(raw: str) -> list[int]:
    """Accept '[0,1,2]', '0 1 2' or '0,1,2'."""
    raw = raw.strip()
    values = ast.literal_eval(raw) if raw.startswith("[") else [t for t in raw.replace(",", " ").split() if t]
    indices = [int(v) for v in values]
    if not indices or len(indices) % 3 != 0:
        raise ValueError(f"Expected a non-empty index list with length a multiple of 3, got {len(indices)}")
    return indices


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def find_gt_depth(depth_dir: Path | None, frame_id: str) -> str | None:
    """Locate a ground-truth depth .npy for `frame_id`, allowing a filename prefix."""
    if depth_dir is None or not depth_dir.is_dir():
        return None
    exact = depth_dir / f"{frame_id}.npy"
    if exact.is_file():
        return str(exact)
    matches = sorted(depth_dir.glob(f"*{frame_id}.npy"))
    return str(matches[0]) if matches else None


def build_triplet_scene(
    dataset: str,
    triplet_id: int,
    indices: list[int],
    data_root: Path,
    scale: float,
    offset: int,
    intrinsics: np.ndarray,
    poses: dict[str, np.ndarray],
    image_files: list[Path],
    depth_dir: Path | None,
) -> tuple[str, list[str]]:
    """Write one triplet as a custom_images scene. Returns (scene_name, warnings).

    `indices` are relative to `offset` within the sorted image list.
    """
    for i in indices:
        if not 0 <= offset + i < len(image_files):
            raise IndexError(
                f"Index {i} (offset {offset}) out of range for {dataset} ({len(image_files)} images)"
            )

    scene_name = f"{dataset}_off{offset:04d}_t{triplet_id:03d}_" + "-".join(str(i) for i in indices)
    scene_dir = data_root / "test" / scene_name
    if scene_dir.exists():
        shutil.rmtree(scene_dir)
    scene_dir.mkdir(parents=True)

    anchor_id = image_files[offset + indices[0]].stem
    if anchor_id not in poses:
        raise KeyError(f"No pose for anchor frame {anchor_id} (index {indices[0]}) in {dataset}")
    T0_inv = se3_inverse(poses[anchor_id])

    warnings: list[str] = []
    frames = []
    for slot, index in enumerate(indices):
        src_index = offset + index
        img_path = image_files[src_index]
        frame_id = img_path.stem
        if frame_id not in poses:
            warnings.append(f"{scene_name}: no pose for frame {frame_id} (index {index}), skipped")
            continue

        c2w = T0_inv @ poses[frame_id]
        if scale != 1.0:
            c2w[:3, 3] /= scale

        out_name = f"{slot:03d}.png"
        shutil.copy2(img_path, scene_dir / out_name)
        frame = {
            "image": out_name,
            "intrinsics": intrinsics.tolist(),
            "extrinsics": c2w.tolist(),
            "index": index,
            "source_index": src_index,
            "frame_id": frame_id,
        }
        gt_depth = find_gt_depth(depth_dir, frame_id)
        if gt_depth is not None:
            frame["depth_gt"] = gt_depth
        frames.append(frame)

    # Loose tolerance: UTM translations are ~1e6, so the relative transform leaves mm-level residuals.
    if not np.allclose(np.array(frames[0]["extrinsics"]), np.eye(4), atol=1e-4):
        raise AssertionError(f"{scene_name}: first frame pose is not identity after relative transform")

    with (scene_dir / "cameras.json").open("w") as f:
        json.dump(
            {
                "scene": scene_name,
                "dataset": dataset,
                "scale": scale,
                "offset": offset,
                "indices": indices,
                "frames": frames,
            },
            f,
            indent=2,
        )

    return scene_name, warnings


def preprocess(
    dataset: str,
    indices: list[int],
    data_root: Path,
    scale: float,
    offset: int,
) -> tuple[list[tuple[int, list[int], str]], list[str]]:
    source_dir = CUSTOM_ROOT / dataset
    intrinsics = load_intrinsics(source_dir / "intrinsics.txt")
    poses = load_trajectory(source_dir / "kf_traj.txt")
    image_files = sorted((source_dir / "imgs").glob("*.png"))
    depth_dir = next((source_dir / n for n in GT_DEPTH_DIR_NAMES if (source_dir / n).is_dir()), None)
    if depth_dir is None:
        print(f"  No ground-truth depth folder in {source_dir} (looked for {'/'.join(GT_DEPTH_DIR_NAMES)})")
    else:
        print(f"  Ground-truth depth folder: {depth_dir}")

    if (data_root / "test").exists():
        shutil.rmtree(data_root / "test")

    rows: list[tuple[int, list[int], str]] = []
    all_warnings: list[str] = []
    for k in range(0, len(indices), 3):
        triplet = indices[k : k + 3]
        scene_name, warnings = build_triplet_scene(
            dataset, k // 3, triplet, data_root, scale, offset, intrinsics, poses, image_files, depth_dir
        )
        all_warnings.extend(warnings)
        rows.append((k // 3, triplet, scene_name))
        print(f"  [{k // 3:03d}] {triplet} (+{offset}) -> {scene_name}")

    return rows, all_warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--indices", help="Flat index list relative to --offset, e.g. '[0,1,2,3,4,5]'")
    source.add_argument("--count", type=int, help="Use consecutive indices range(count); must be a multiple of 3")
    parser.add_argument("--dataset", default="realm_1", choices=["realm_1", "realm_2", "realm_1_with_depth"])
    parser.add_argument("--offset", type=int, default=0, help="Index of the first frame; indices are relative to it")
    parser.add_argument("--scale", type=float, default=1.0, help="Divide translations by this factor")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-inference", action="store_true", help="Only run pre-processing")
    parser.add_argument(
        "--fp32",
        choices=["on", "off"],
        default="on",
        help="on (default): full fp32, so the depth maps written here can be used as "
        "a numerical reference for a compiled AOTInductor build. off: allow TF32, "
        "~19%% faster but ~1.3 m different. See src/precision.py.",
    )
    parser.add_argument(
        "--xformers",
        choices=XFORMERS_MODES,
        default="auto",
        help="on: force xformers; off: force the exportable SDPA fallback; "
        "auto: inherit XFORMERS_DISABLED from the environment (default). "
        "Applied before imports, so the value here is authoritative.",
    )
    args = parser.parse_args()

    if args.count is not None:
        if args.count <= 0 or args.count % 3 != 0:
            parser.error(f"--count must be a positive multiple of 3, got {args.count}")
        indices = list(range(args.count))
    else:
        indices = parse_indices(args.indices)
    output_dir = args.output_dir or REPO_ROOT / "outputs" / f"depthsplat-depth-base-{args.dataset}-triplets"

    print_xformers_state("startup")

    rows, warnings = preprocess(args.dataset, indices, args.data_root, args.scale, args.offset)
    for warning in warnings:
        print(f"  WARNING: {warning}")

    if args.skip_inference:
        return 0

    dataset_cfg = yaml.safe_load(DATASET_CFG_PATH.read_text())
    cfg = compose_config(
        depth_inference_overrides(
            dataset_root=args.data_root.relative_to(REPO_ROOT),
            output_dir=output_dir.relative_to(REPO_ROOT),
            image_shape=tuple(dataset_cfg["image_shape"]),
            near=dataset_cfg["near"],
            far=dataset_cfg["far"],
        )
    )
    run_depth_inference(cfg, full_fp32=args.fp32 == "on")

    print_xformers_state("after inference")

    print("\ntriplet | indices | scene | output | status")
    for k, triplet, scene_name in rows:
        scene_out = output_dir / "images" / scene_name / "depth"
        status = "ok" if scene_out.is_dir() else "missing"
        print(f"{k:>7} | {triplet} | {scene_name} | {scene_out} | {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
