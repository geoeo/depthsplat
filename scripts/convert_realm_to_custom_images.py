#!/usr/bin/env python3
"""Convert realm_1 and realm_2 datasets to custom_images format."""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    """Load intrinsics from intrinsics.txt (fx fy cx cy width height)."""
    with open(intrinsics_path, "r") as f:
        line = f.readline().strip()
    
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
    Assumes W2C (world-to-camera) format; returns as C2W (camera-to-world).
    """
    poses = {}
    with open(traj_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            frame_id = parts[0]
            matrix_vals = np.array(list(map(float, parts[1:])), dtype=float)
            
            # Construct 3x4 W2C matrix from the 12 values
            w2c = matrix_vals.reshape(3, 4)
            
            # Pad to 4x4 and invert to get C2W
            w2c_4x4 = np.vstack([w2c, [0, 0, 0, 1]])
            c2w_4x4 = np.linalg.inv(w2c_4x4)
            
            poses[frame_id] = c2w_4x4
    
    return poses


def convert_dataset(realm_name: str) -> None:
    """Convert a realm dataset to custom_images format."""
    source_dir = Path("/workspaces/custom") / realm_name
    output_dir = Path("/workspaces/datasets/custom_images/test") / realm_name
    
    print(f"Converting {realm_name}...")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load intrinsics and trajectory
    intrinsics = load_intrinsics(source_dir / "intrinsics.txt")
    poses = load_trajectory(source_dir / "kf_traj.txt")
    
    # Get sorted image files
    img_dir = source_dir / "imgs"
    image_files = sorted(img_dir.glob("*.png"))
    
    if not image_files:
        print(f"  WARNING: No images found in {img_dir}")
        return
    
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
        c2w = poses[frame_id].tolist()
        extrinsics = [c2w[i][:4] for i in range(3)]  # 3x4 matrix
        
        frames.append({
            "image": output_img_name,
            "intrinsics": intrinsics.tolist(),
            "extrinsics": extrinsics,
        })
    
    # Write cameras.json
    cameras_data = {
        "scene": realm_name,
        "frames": frames,
    }
    
    cameras_json_path = output_dir / "cameras.json"
    with open(cameras_json_path, "w") as f:
        json.dump(cameras_data, f, indent=2)
    
    print(f"  ✓ Converted {len(frames)} frames")
    print(f"  ✓ Saved to {output_dir}")


if __name__ == "__main__":
    for realm_name in ["realm_1", "realm_2"]:
        convert_dataset(realm_name)
    print("\nConversion complete!")
