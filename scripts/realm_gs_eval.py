#!/usr/bin/env python3
"""Gaussian-splatting evaluation run on realm windows, with a manifest and tables.

Runs the full DepthSplat model in-process (full fp32 by default) on the scenes
of a custom_images root using the given evaluation index, then writes
<output_dir>/manifest.json, summary.md / summary.json (per-scene, per-view
PSNR / SSIM / LPIPS) and, optionally, the depth-consistency and DEM reports.

    python scripts/realm_gs_eval.py --data-root datasets/custom_images_gs \
        --index-path assets/evaluation_index_realm_gs.json \
        --output-dir outputs/gs-eval-realm_2_with_depth --gaussian-scale-max 1.0

image_shape, near and far come from config/dataset/custom_images.yaml through
+experiment=realm and apply to the whole scene; they are recorded in the manifest
but cannot be set here.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import-time flags, must be set before the first `src` import (see realm_triplet_depth.py).
_XFORMERS_MODES = ("auto", "on", "off")


def _preparse(flag: str, argv: list[str]) -> str | None:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


_xf = _preparse("--xformers", sys.argv[1:]) or "auto"
if _xf not in _XFORMERS_MODES:
    raise SystemExit(f"--xformers must be one of {_XFORMERS_MODES}, got {_xf!r}")
if _xf == "off":
    os.environ["XFORMERS_DISABLED"] = "1"
elif _xf == "on":
    os.environ.pop("XFORMERS_DISABLED", None)

import yaml  # noqa: E402

from gs_scores_table import summarize  # noqa: E402
from src.gs_inference import DEFAULT_PRETRAINED_GS, compose_config, gs_inference_overrides, run_gs_inference  # noqa: E402

DATASET_CFG_PATH = REPO_ROOT / "config" / "dataset" / "custom_images.yaml"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "datasets" / "custom_images_gs")
    parser.add_argument("--index-path", type=Path, default=REPO_ROOT / "assets" / "evaluation_index_realm_gs.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", default=None, help="Only this scene (default: every scene in the root that is in the index)")
    parser.add_argument("--checkpoint", default=DEFAULT_PRETRAINED_GS)
    parser.add_argument("--depth-checkpoint", default=None,
                        help="Hybrid: overwrite the depth predictor with this depth-only checkpoint after loading"
                             " --checkpoint (metric depth; use with --no-scene-scale-normalization)")
    parser.add_argument("--vit", choices=["vits", "vitb"], default="vitb")
    parser.add_argument("--num-scales", type=int, default=2)
    parser.add_argument("--upsample-factor", type=int, default=4)
    parser.add_argument("--lowest-feature-resolution", type=int, default=8)
    parser.add_argument("--gaussian-scale-max", type=float, default=0.1,
                        help="W3 knob. In normalised units (sqrt(near*far) = --scene-scale-target-depth) when"
                             " the scene-scale normalisation is on (default), metres when it is off")
    parser.add_argument("--no-scene-scale-normalization", action="store_true",
                        help="Feed metric cameras to the network unscaled (the GS checkpoints then collapse"
                             " to the far plane; for A/B checks only)")
    parser.add_argument("--scene-scale-mode", choices=["nearfar", "baseline"], default="nearfar",
                        help="nearfar (default): sqrt(near*far) of the yaml -> --scene-scale-target-depth units;"
                             " baseline: mean consecutive context baseline -> --scene-scale-baseline")
    parser.add_argument("--scene-scale-target-depth", type=float, default=3.0)
    parser.add_argument("--scene-scale-baseline", type=float, default=1.0)
    parser.add_argument("--fp32", choices=["on", "off"], default="on", help="on: full fp32 (default); off: TF32 like src.main")
    parser.add_argument("--xformers", choices=_XFORMERS_MODES, default="auto")
    parser.add_argument("--no-gaussian", action="store_true", help="Skip the PLY export")
    parser.add_argument("--consistency", action="store_true", help="Also run scripts/depth_consistency.py on the encoder depth")
    parser.add_argument("--dem-eval", action="store_true", help="Also compare encoder and rendered depth with the OpenREALM reference DEM (scripts/realm_dem_eval.py; a reference from an earlier reconstruction, not ground truth)")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    index_path = args.index_path.resolve()
    output_dir = args.output_dir.resolve()
    if not index_path.is_file():
        parser.error(f"index not found: {index_path}")
    index = json.loads(index_path.read_text())
    scenes = sorted(p.name for p in (data_root / "test").iterdir() if (p / "cameras.json").is_file() and p.name in index)
    if args.scene:
        scenes = [s for s in scenes if s == args.scene]
    if not scenes:
        parser.error(f"no scene under {data_root}/test is listed in {index_path}")

    dataset_cfg = yaml.safe_load(DATASET_CFG_PATH.read_text())
    manifest = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "data_root": str(data_root),
        "index_path": str(index_path),
        "scenes": {s: index[s] for s in scenes},
        "image_shape": dataset_cfg["image_shape"],
        "near": dataset_cfg["near"],
        "far": dataset_cfg["far"],
        "near_far_source": str(DATASET_CFG_PATH.relative_to(REPO_ROOT)),
        "checkpoint": args.checkpoint,
        "depth_checkpoint": args.depth_checkpoint,
        "vit": args.vit,
        "num_scales": args.num_scales,
        "upsample_factor": args.upsample_factor,
        "lowest_feature_resolution": args.lowest_feature_resolution,
        "gaussian_scale_max": args.gaussian_scale_max,
        "normalize_scene_scale": not args.no_scene_scale_normalization,
        "scene_scale_mode": args.scene_scale_mode,
        "scene_scale_target_depth": args.scene_scale_target_depth,
        "scene_scale_baseline": args.scene_scale_baseline,
        "precision": "fp32" if args.fp32 == "on" else "tf32",
        "xformers_mode": args.xformers,
        "XFORMERS_DISABLED": os.environ.get("XFORMERS_DISABLED"),
    }

    cfg = compose_config(
        gs_inference_overrides(
            dataset_root=data_root,
            output_dir=output_dir,
            index_path=index_path,
            scene_selection=args.scene,
            pretrained_model=args.checkpoint,
            pretrained_depth=args.depth_checkpoint,
            vit_type=args.vit,
            num_scales=args.num_scales,
            upsample_factor=args.upsample_factor,
            lowest_feature_resolution=args.lowest_feature_resolution,
            gaussian_scale_max=args.gaussian_scale_max,
            normalize_scene_scale=not args.no_scene_scale_normalization,
            scene_scale_mode=args.scene_scale_mode,
            scene_scale_target_depth=args.scene_scale_target_depth,
            scene_scale_baseline=args.scene_scale_baseline,
            save_gaussian=not args.no_gaussian,
        )
    )
    manifest["near_far_composed"] = [float(cfg.dataset.near), float(cfg.dataset.far)]
    import torch  # noqa: E402  (after the import-time flags)

    manifest["torch"] = torch.__version__
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    run_gs_inference(cfg, full_fp32=args.fp32 == "on")

    summarize(output_dir, manifest)
    print((output_dir / "summary.md").read_text())

    py = sys.executable
    if args.consistency:
        subprocess.run([py, str(REPO_ROOT / "scripts" / "depth_consistency.py"), "--run-dir", str(output_dir),
                        "--data-root", str(data_root), "--source", "encoder"], check=False)
    if args.dem_eval:
        for source in ("encoder", "rendered"):
            subprocess.run([py, str(REPO_ROOT / "scripts" / "realm_dem_eval.py"), "--run-dir", str(output_dir),
                            "--data-root", str(data_root), "--source", source], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
