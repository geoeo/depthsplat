#!/usr/bin/env python3
"""Visualize depth .npy files one at a time with matplotlib."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

IMAGES_DIR = Path("/workspaces/outputs/depthsplat-depth-base-realm_1_with_depth-triplets/images")


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
    npy_files = sorted(p for p in IMAGES_DIR.glob("*/depth/*.npy") if not p.stem.endswith("_gt"))
    if not npy_files:
        print(f"No .npy files found under {IMAGES_DIR}")
        return

    print(
        f"Found {len(npy_files)} files under {IMAGES_DIR}. "
        f"Using scale={args.scale}. Press Enter to advance, Ctrl+C to quit."
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.subplots_adjust(left=0.04, right=0.95, wspace=0.45)
    plt.ion()
    # Fixed divider axes keep all three image axes identically proportioned; the first
    # one is unused but reserves the same width as the colorbars.
    caxes = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05) for ax in axes]
    caxes[0].set_axis_off()

    for path in npy_files:
        depth = np.load(path) * args.scale
        gt_path = path.with_name(f"{path.stem}_gt.npy")
        gt = np.load(gt_path) * args.scale if gt_path.is_file() else None
        scene = path.parent.parent.name
        label = f"{scene}/{path.name}"

        rgb_path = path.with_name(f"{path.stem}_cam.png")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB")) if rgb_path.is_file() else None

        for ax in axes:
            ax.clear()
            ax.set_visible(False)
        for cax in caxes[1:]:
            cax.clear()
            cax.set_visible(False)

        if rgb is not None:
            axes[0].set_visible(True)
            axes[0].imshow(rgb, aspect="equal")
            axes[0].set_title(f"{scene}/{rgb_path.name}")
            axes[0].set_xlabel("x (pixels)")
            axes[0].set_ylabel("y (pixels)")

        panels = [(1, depth, label)]
        if gt is not None:
            panels.append((2, gt, f"{label} (gt)"))

        for i, data, title in panels:
            ax = axes[i]
            ax.set_visible(True)
            img = ax.imshow(data, cmap="plasma", aspect="equal")
            caxes[i].set_visible(True)
            fig.colorbar(img, cax=caxes[i], label="Depth (m)")
            ax.set_title(title)
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")

        fig.canvas.draw()
        plt.pause(0.01)
        summary = f"[{label}]  min={depth.min():.2f}m  max={depth.max():.2f}m"
        if gt is not None:
            summary += f"  | gt min={gt.min():.2f}m  max={gt.max():.2f}m"
        input(f"{summary} — press Enter for next ")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
