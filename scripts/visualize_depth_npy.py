#!/usr/bin/env python3
"""Visualize depth .npy files one at a time with matplotlib."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEPTH_DIR = "/workspaces/outputs/depthsplat-depth-base-realm_1-triplets/images/realm_1_off0070_t015_45-46-47/depth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply loaded depth values by this factor before visualization",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    npy_files = sorted(Path(DEPTH_DIR).glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {DEPTH_DIR}")
        return

    print(
        f"Found {len(npy_files)} files in {DEPTH_DIR}. "
        f"Using scale={args.scale}. Press Enter to advance, Ctrl+C to quit."
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()

    for path in npy_files:
        depth = np.load(path) * args.scale

        ax.clear()
        img = ax.imshow(depth, cmap="plasma")

        if not hasattr(main, "_cbar"):
            main._cbar = fig.colorbar(img, ax=ax, label="Depth (m)")
        else:
            main._cbar.update_normal(img)

        ax.set_title(path.name)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        fig.canvas.draw()
        plt.pause(0.01)
        input(f"[{path.name}]  min={depth.min():.2f}m  max={depth.max():.2f}m  — press Enter for next ")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
