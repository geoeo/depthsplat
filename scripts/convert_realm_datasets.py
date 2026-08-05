#!/usr/bin/env python3
"""
Convert realm_1 and realm_2 datasets to custom_images format.

Input format:
  custom/realm_X/
    intrinsics.txt: fx fy cx cy width height
    kf_traj.txt: ts r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz (C2W)
    imgs/: image files (000.png, 001.png, ...)

Output format:
  datasets/custom_images/test/realm_X/
    000.png, 001.png, ...
    cameras.json: scene metadata with intrinsics and extrinsics
"""

import json
import shutil
from pathlib import Path
import numpy as np


def convert_dataset(dataset_name: str, src_dir: Path, dst_dir: Path) -> None:
    """Convert a realm dataset to custom_images format."""
    
    # Read intrinsics
    intrinsics_file = src_dir / "intrinsics.txt"
    with open(intrinsics_file) as f:
        # Skip comment line
        f.readline()
        fx, fy, cx, cy, width, height = map(float, f.readline().split())
    
    # Read keyframe trajectory
    kf_traj_file = src_dir / "kf_traj.txt"
    frames = []
    image_dir = src_dir / "imgs"
    
    with open(kf_traj_file) as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            ts_str = parts[0]  # Keep as string to avoid float precision loss
            
            # Extract 3x4 C2W matrix: r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz
            extrinsics_flat = list(map(float, parts[1:13]))
            extrinsics = np.array([
                extrinsics_flat[0:4],
                extrinsics_flat[4:8],
                extrinsics_flat[8:12],
                [0, 0, 0, 1]
            ]).tolist()
            
            # Image name matches the timestamp from kf_traj.txt
            image_name = f"{ts_str}.png"
            image_path = image_dir / image_name
            
            if image_path.exists():
                # Rename to sequential for the output
                output_image_name = f"{line_idx:03d}.png"
                frames.append({
                    "image": output_image_name,
                    "intrinsics": [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ],
                    "extrinsics": extrinsics,
                    "_src_image": image_name  # Keep track of source for copying
                })
    
    # Create output directory
    scene_dir = dst_dir / dataset_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    print(f"Copying {len(frames)} images for {dataset_name}...")
    for frame in frames:
        src_img = image_dir / frame["_src_image"]
        dst_img = scene_dir / frame["image"]
        shutil.copy(src_img, dst_img)
        del frame["_src_image"]
    
    # Write cameras.json
    metadata = {
        "scene": dataset_name,
        "frames": frames
    }
    
    metadata_file = scene_dir / "cameras.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(frames)} frames to {metadata_file}")


def main():
    workspace_root = Path(__file__).parent.parent
    src_base = workspace_root / "custom"
    dst_base = workspace_root / "datasets" / "custom_images" / "test"
    
    for dataset_name in ["realm_1", "realm_2"]:
        src_dir = src_base / dataset_name
        if src_dir.exists():
            print(f"\nConverting {dataset_name}...")
            convert_dataset(dataset_name, src_dir, dst_base)
        else:
            print(f"Warning: {src_dir} not found")


if __name__ == "__main__":
    main()
