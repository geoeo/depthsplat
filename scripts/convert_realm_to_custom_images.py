#!/usr/bin/env python3
"""Convert realm_1 and realm_2 datasets to custom_images format."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    """Load intrinsics from intrinsics.txt (fx fy cx cy width height)."""
    with open(intrinsics_path, "r") as f:
        line = next(l for l in f if not l.startswith("#")).strip()

    values = list(map(float, line.split()))
    fx, fy, cx, cy = values[:4]
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=float)
    return K


def load_trajectory(traj_path: Path) -> Dict[str, np.ndarray]:
    """Load camera trajectory from kf_traj.txt.

    Format: frame_id R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz
    Translation column holds the UTM camera position, so this is C2W.
    """
    poses = {}
    with open(traj_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            frame_id = parts[0]
            matrix_vals = np.array(list(map(float, parts[1:])), dtype=float)

            # Construct 4x4 C2W matrix directly (no inversion needed)
            c2w = matrix_vals.reshape(3, 4)
            c2w_4x4 = np.vstack([c2w, [0, 0, 0, 1]])

            poses[frame_id] = c2w_4x4

    return poses


def convert_dataset(realm_name: str, offset: int = 0, num_frames: int | None = None, scale: float = 1.0) -> None:
    """Convert a realm dataset to custom_images format."""
    source_dir = Path("/workspaces/custom") / realm_name
    output_dir = Path("/workspaces/datasets/custom_images/test") / realm_name
    
    print(f"Converting {realm_name}...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    # Load intrinsics and trajectory
    intrinsics = load_intrinsics(source_dir / "intrinsics.txt")
    poses = load_trajectory(source_dir / "kf_traj.txt")

    # Get sorted image files
    img_dir = source_dir / "imgs"
    image_files = sorted(img_dir.glob("*.png"))

    if not image_files:
        print(f"  WARNING: No images found in {img_dir}")
        return

    image_files = image_files[offset : (offset + num_frames) if num_frames is not None else None]
    print(f"  Frames: offset={offset}, count={len(image_files)}, scale={scale}")

    # Express all poses relative to the first selected frame.
    frame_ids_sorted = sorted(poses.keys())
    T0 = poses[frame_ids_sorted[offset]]
    R0, t0 = T0[:3, :3], T0[:3, 3]
    # SE(3) inverse: [R^T | -R^T t]
    T0_inv = np.eye(4)
    T0_inv[:3, :3] = R0.T
    T0_inv[:3, 3] = -R0.T @ t0
    for frame_id in poses:
        poses[frame_id] = T0_inv @ poses[frame_id]

    if scale != 1.0:
        for frame_id in poses:
            poses[frame_id][:3, 3] /= scale

    # Create frame list
    frames = []
    for idx, img_path in enumerate(image_files):
        frame_id = img_path.stem
        
        if frame_id not in poses:
            print(f"  WARNING: No pose for frame {frame_id}, skipping")
            continue
        
        # Create zero-padded filename
        output_img_name = f"{idx:03d}.png"
        output_img_path = output_dir / output_img_name
        
        # Copy image
        shutil.copy2(img_path, output_img_path)
        
        # Get C2W pose as list of lists
        extrinsics = poses[frame_id].tolist()  # 4x4 matrix
        
        frames.append({
            "image": output_img_name,
            "intrinsics": intrinsics.tolist(),
            "extrinsics": extrinsics,
        })
    
    # Write cameras.json
    cameras_data = {
        "scene": realm_name,
        "scale": scale,
        "frames": frames,
    }

    cameras_json_path = output_dir / "cameras.json"
    with open(cameras_json_path, "w") as f:
        json.dump(cameras_data, f, indent=2)

    print(f"  ✓ Converted {len(frames)} frames")
    print(f"  ✓ Saved to {output_dir}")

    return realm_name, len(frames)


def build_eval_index(
    scene_frame_counts: dict[str, int],
    index_path: Path,
    context_step_size: int,
) -> None:
    """Build an evaluation index JSON for the converted realm scenes.

    Context: every `context_step_size`-th frame. Target: all frames.
    """
    if context_step_size <= 0:
        raise ValueError("context_step_size must be greater than 0")

    index = {}
    for scene, n in scene_frame_counts.items():
        all_indices = list(range(n))
        context = all_indices[::context_step_size]
        target = all_indices
        index[scene] = {"context": context, "target": target}

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n✓ Evaluation index saved to {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0, help="Index of the first frame to include")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames to include after offset")
    parser.add_argument("--scale", type=float, default=1.0, help="Divide all translations by this factor to fit scene into near/far range")
    parser.add_argument("--context-step-size", type=int, default=1, help="Use every Nth frame as a context view")
    args = parser.parse_args()

    scene_frame_counts = {}
    for realm_name in ["realm_1", "realm_2"]:
        scene, n = convert_dataset(realm_name, offset=args.offset, num_frames=args.num_frames, scale=args.scale)
        scene_frame_counts[scene] = n

    build_eval_index(
        scene_frame_counts,
        Path("assets/evaluation_index_realm.json"),
        args.context_step_size,
    )
    print("\nConversion complete!")
