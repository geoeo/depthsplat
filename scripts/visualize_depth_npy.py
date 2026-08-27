#!/usr/bin/env python3
"""Visualize depth .npy files one at a time with matplotlib."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEPTH_DIR = "/workspaces/outputs/depthsplat-depth-base-realm_1_with_depth-triplets/images/realm_1_with_depth_off0070_t009_27-28-29/depth"


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
    npy_files = sorted(p for p in Path(DEPTH_DIR).glob("*.npy") if not p.stem.endswith("_gt"))
    if not npy_files:
        print(f"No .npy files found in {DEPTH_DIR}")
        return

    print(
        f"Found {len(npy_files)} files in {DEPTH_DIR}. "
        f"Using scale={args.scale}. Press Enter to advance, Ctrl+C to quit."
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.ion()
    colorbars = []

    for path in npy_files:
        depth = np.load(path) * args.scale
        gt_path = path.with_name(f"{path.stem}_gt.npy")
        gt = np.load(gt_path) * args.scale if gt_path.is_file() else None

        for cbar in colorbars:
            cbar.remove()
        colorbars = []

        panels = [(axes[0], depth, path.name)]
        if gt is not None:
            panels.append((axes[1], gt, gt_path.name))
            axes[1].set_visible(True)
        else:
            axes[1].clear()
            axes[1].set_visible(False)

        for ax, data, title in panels:
            ax.clear()
            img = ax.imshow(data, cmap="plasma")
            colorbars.append(fig.colorbar(img, ax=ax, label="Depth (m)"))
            ax.set_title(title)
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")

        fig.canvas.draw()
        plt.pause(0.01)
        summary = f"[{path.name}]  min={depth.min():.2f}m  max={depth.max():.2f}m"
        if gt is not None:
            summary += f"  | gt min={gt.min():.2f}m  max={gt.max():.2f}m"
        input(f"{summary} — press Enter for next ")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
