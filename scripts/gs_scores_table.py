#!/usr/bin/env python3
"""Per-scene, per-view novel-view metrics for a Gaussian-splatting test run.

Recomputes PSNR / SSIM / LPIPS from the images the run saved
(images/<scene>/color/<idx>.png next to <idx>_gt.png) with the same functions
src.main uses, so the numbers do not depend on dataloader order and come with
scene names and target indices. Writes <run_dir>/summary.md and summary.json;
the run's own averaged scores (metrics/scores_all_avg.json) are included for a
cross-check when present.

    python scripts/gs_scores_table.py outputs/depthsplat-gs-base-realm_2-smoke
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim  # noqa: E402

GT_PATTERN = re.compile(r"^(\d+)_gt\.png$")


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device)  # [1, 3, H, W]


def score_run(run_dir: Path, device: torch.device) -> dict:
    scenes = {}
    for gt_path in sorted(run_dir.glob("images/*/color/*_gt.png")):
        index = int(GT_PATTERN.match(gt_path.name).group(1))
        pred_path = gt_path.with_name(f"{index:0>6}.png")
        if not pred_path.is_file():
            continue
        gt = load_image(gt_path, device)
        pred = load_image(pred_path, device)
        scene = gt_path.parent.parent.name
        scenes.setdefault(scene, []).append({
            "target_index": index,
            "psnr": compute_psnr(gt, pred).item(),
            "ssim": compute_ssim(gt, pred).item(),
            "lpips": compute_lpips(gt, pred).item(),
        })

    per_scene = {}
    for scene, views in scenes.items():
        per_scene[scene] = {
            "views": views,
            **{m: float(np.mean([v[m] for v in views])) for m in ("psnr", "ssim", "lpips")},
        }
    overall = {}
    for m in ("psnr", "ssim", "lpips"):
        values = [v[m] for s in scenes.values() for v in s]
        overall[m] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n_views": len(values)} if values else None
    return {"per_scene": per_scene, "overall": overall}


def render_markdown(result: dict, run_dir: Path, run_scores: dict | None, manifest: dict | None) -> str:
    lines = [f"# Scores for `{run_dir}`", ""]
    if manifest:
        lines.append("| setting | value |")
        lines.append("|---|---|")
        for k, v in manifest.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    ov = result["overall"]
    if ov["psnr"] is None:
        lines.append("No rendered/ground-truth image pairs found under images/*/color/.")
        return "\n".join(lines) + "\n"
    lines.append("| scene | targets | PSNR (dB) | SSIM | LPIPS |")
    lines.append("|---|---|---|---|---|")
    for scene, s in result["per_scene"].items():
        targets = ",".join(str(v["target_index"]) for v in s["views"])
        lines.append(f"| {scene} | {targets} | {s['psnr']:.2f} | {s['ssim']:.3f} | {s['lpips']:.3f} |")
    lines.append(
        f"| **all ({ov['psnr']['n_views']} views)** | | **{ov['psnr']['mean']:.2f} ± {ov['psnr']['std']:.2f}** "
        f"| **{ov['ssim']['mean']:.3f} ± {ov['ssim']['std']:.3f}** | **{ov['lpips']['mean']:.3f} ± {ov['lpips']['std']:.3f}** |"
    )
    lines.append("")
    lines.append("Per view:")
    lines.append("")
    lines.append("| scene | target | PSNR (dB) | SSIM | LPIPS |")
    lines.append("|---|---|---|---|---|")
    for scene, s in result["per_scene"].items():
        for v in s["views"]:
            lines.append(f"| {scene} | {v['target_index']} | {v['psnr']:.2f} | {v['ssim']:.3f} | {v['lpips']:.3f} |")
    if run_scores:
        lines.append("")
        lines.append("Run's own batch-averaged scores (`metrics/scores_all_avg.json`): "
                     + ", ".join(f"{k}={v:.4f}" for k, v in run_scores.items() if isinstance(v, float)))
    return "\n".join(lines) + "\n"


def summarize(run_dir: Path, manifest: dict | None = None, device: str | None = None) -> dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = score_run(run_dir, device_t)
    avg_path = run_dir / "metrics" / "scores_all_avg.json"
    run_scores = json.loads(avg_path.read_text()) if avg_path.is_file() else None
    result["run_scores"] = run_scores
    result["manifest"] = manifest
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2))
    markdown = render_markdown(result, run_dir, run_scores, manifest)
    (run_dir / "summary.md").write_text(markdown)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manifest", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Extra settings to record in the summary header")
    args = parser.parse_args()
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    for kv in args.manifest:
        k, _, v = kv.partition("=")
        manifest[k] = v
    summarize(args.run_dir, manifest or None, args.device)
    print((args.run_dir / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
