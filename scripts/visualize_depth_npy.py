#!/usr/bin/env python3
"""Visualize depth .npy files one at a time with matplotlib.

Handles both layouts this repo produces:

  * the pipeline (realm_triplet_depth.py) writes one 2-D [H, W] file per view,
    at <out>/images/<scene>/depth/000000.npy, alongside 000000_gt.npy and
    000000_cam.png;
  * run_depth_aoti.py writes one file per scene, <out>/<scene>.npy, holding the
    whole batch as [B, V, H, W] with no ground truth or camera image.

Ground truth, the eager-mode depth, and the camera image are shown when they are
found next to the depth file, and simply omitted when they are not -- a
scene-level file from run_depth_aoti.py renders as a single depth panel.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

DEFAULT_DIR = Path("/workspaces/outputs/depthsplat-depth-base-realm_1_s")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"directory to search recursively for depth .npy files (default: {DEFAULT_DIR})",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply loaded depth values by this factor before visualization",
    )
    parser.add_argument(
        "--display-eager",
        action="store_true",
        help="also show the eager-mode depth panel when a `..._eager.npy` sibling exists",
    )
    return parser.parse_args()


def find_depth_files(root: Path) -> list[Path]:
    """Every depth .npy under `root`, in either layout, ground truth/eager excluded.

    run_depth_aoti.py also writes a sibling `..._eager.npy` per view; without excluding
    it here it gets treated as its own primary file, whose `..._eager_cam.png`
    lookup then misses even though the real `..._cam.png` exists.
    """
    return sorted(
        p for p in root.rglob("*.npy")
        if not p.stem.endswith("_gt") and not p.stem.endswith("_eager")
    )


def format_range(data: np.ndarray) -> str:
    """min/max ignoring NaN, which the ground-truth maps use for holes."""
    if not np.isfinite(data).any():
        return "all NaN"
    return f"min={np.nanmin(data):.2f}m  max={np.nanmax(data):.2f}m"


def iter_frames(array: np.ndarray, label: str):
    """Yield (title, 2-D depth) for a file that may hold one view or many.

    [H, W] -> one frame; [V, H, W] and [B, V, H, W] -> one frame per view.
    """
    if array.ndim == 2:
        yield label, array
    elif array.ndim == 3:
        for v in range(array.shape[0]):
            yield f"{label} [v{v}]", array[v]
    elif array.ndim == 4:
        b_count = array.shape[0]
        for b in range(b_count):
            for v in range(array.shape[1]):
                tag = f"[v{v}]" if b_count == 1 else f"[b{b} v{v}]"
                yield f"{label} {tag}", array[b, v]
    else:
        print(f"  skipping {label}: unsupported shape {array.shape}")


def main():
    args = parse_args()
    root = args.input_dir
    if not root.is_dir():
        print(f"No such directory: {root}")
        return

    npy_files = find_depth_files(root)
    if not npy_files:
        print(f"No .npy files found under {root}")
        return

    print(
        f"Found {len(npy_files)} file(s) under {root}. "
        f"Using scale={args.scale}. Press Enter to advance, Ctrl+C to quit."
    )

    plt.ion()
    fig = None
    axes = []
    caxes = []

    for path in npy_files:
        array = np.load(path) * args.scale
        # Siblings exist only in the pipeline layout; absent for scene-level files.
        gt_path = path.with_name(f"{path.stem}_gt.npy")
        gt = np.load(gt_path) * args.scale if gt_path.is_file() else None
        eager_path = path.with_name(f"{path.stem}_eager.npy")
        eager = np.load(eager_path) * args.scale if args.display_eager and eager_path.is_file() else None
        rgb_path = path.with_name(f"{path.stem}_cam.png")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB")) if rgb_path.is_file() else None

        scene = path.parent.parent.name if path.parent.name == "depth" else path.stem
        base = f"{scene}/{path.name}" if path.parent.name == "depth" else scene

        for title, depth in iter_frames(array, base):
            # Left to right: camera image, depth, ground truth, eager -- each only if present.
            panels = []
            if rgb is not None:
                panels.append((rgb, f"{scene}/{rgb_path.name}", None, False))
            panels.append((depth, title, "plasma", True))
            if gt is not None and gt.shape == depth.shape:
                panels.append((gt, f"{title} (gt)", "plasma", True))
            if eager is not None and eager.shape == depth.shape:
                panels.append((eager, f"{title} (eager)", "plasma", True))

            if len(axes) != len(panels):
                if fig is not None:
                    plt.close(fig)
                fig, subplot_axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
                fig.subplots_adjust(left=0.04, right=0.95, wspace=0.45)
                axes = np.atleast_1d(subplot_axes).tolist()
                # One divider axis per image axis keeps panels identically proportioned.
                caxes = [make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05) for ax in axes]

            for ax, cax in zip(axes, caxes):
                ax.clear()
                ax.set_visible(False)
                cax.clear()
                cax.set_axis_off()

            for i, (data, panel_title, cmap, show_bar) in enumerate(panels):
                ax = axes[i]
                ax.set_visible(True)
                img = ax.imshow(data, cmap=cmap, aspect="equal")
                if show_bar:
                    caxes[i].set_axis_on()
                    fig.colorbar(img, cax=caxes[i], label="Depth (m)")
                ax.set_title(panel_title)
                ax.set_xlabel("x (pixels)")
                ax.set_ylabel("y (pixels)")

            fig.canvas.draw()
            plt.pause(0.01)
            summary = f"[{title}]  {format_range(depth)}"
            if len(panels) > 1 and gt is not None:
                summary += f"  | gt {format_range(gt)}"
            if len(panels) > 1 and eager is not None:
                summary += f"  | eager {format_range(eager)}"
            input(f"{summary} — press Enter for next ")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
