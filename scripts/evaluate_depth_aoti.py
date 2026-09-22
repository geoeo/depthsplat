#!/usr/bin/env python3
"""Diff a compiled depth package against the eager reference stored in a snapshot.

Example:
    python scripts/evaluate_depth_aoti.py \
        --package outputs/aoti/fp32/depth_predictor_fp32.pt2 \
        --dataset-pt2 outputs/dataset_cfg_with_reference_s_500.pt2

The reference is `depth_eager`: what the eager Python model produced for the very
same input tensors, recorded by scripts/export_dataset.py. So the reported
numbers are a plain elementwise difference between two depth maps in metres --
mean and max absolute, RMSE, and the largest relative deviation. There is no
scale alignment and no validity mask to apply: both maps come from the same
model on the same inputs, so every pixel is comparable.

This deliberately does not score against the external depth reference. The
custom/<dataset>/dense .npy maps that earlier snapshots embedded
are unscaled OpenREALM stereo -- "Scaling (Not Georeferenced)" per their sibling
.txt -- and reproject between neighbouring views at r~0.03, against r~0.87 for
the model's own output. Scoring a build against them measures the dataset, not
the compile.

--fp32 must match both how the package was built and how the snapshot's eager
reference was produced; both are checked at startup, because a mismatch shows up
as metres of difference and reads as a broken build.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede torch/src imports; this also matches the compiled runner's setup.
import depth_export_common as common  # noqa: E402

import torch  # noqa: E402


def diff_sums(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Elementwise difference stats for one scene, as running totals."""
    if prediction.shape != reference.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"reference shape {tuple(reference.shape)}"
        )
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite depth")
    if not torch.isfinite(reference).all():
        raise ValueError("eager reference contains non-finite depth")

    prediction = prediction.double()
    reference = reference.double()
    error = (prediction - reference).abs()
    relative = error / reference.abs().clamp(min=1e-6)
    return {
        "count": float(error.numel()),
        "abs_error": error.sum().item(),
        "sq_error": error.square().sum().item(),
        "max_abs": error.max().item(),
        "max_rel": relative.max().item(),
    }


def summarize(sums: dict[str, float]) -> dict[str, float]:
    count = sums["count"]
    return {
        "mean_abs": sums["abs_error"] / count,
        "rmse": (sums["sq_error"] / count) ** 0.5,
        "max_abs": sums["max_abs"],
        "max_rel": sums["max_rel"],
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"mean|d|={metrics['mean_abs']:.3e} m  max|d|={metrics['max_abs']:.3e} m  "
        f"rmse={metrics['rmse']:.3e} m  maxrel={metrics['max_rel']:.3e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True,
                        help="compiled AOTInductor .pt2 package")
    parser.add_argument("--dataset-pt2", type=Path, required=True,
                        help="dataset snapshot containing model inputs and depth_eager")
    parser.add_argument("--device", default="cuda",
                        help="device for dataset tensors; must match the compiled package")
    common.add_precision_arg(parser)
    args = parser.parse_args()

    if not args.package.is_file():
        raise SystemExit(f"no such compiled package: {args.package}")
    if not args.dataset_pt2.is_file():
        raise SystemExit(f"no such dataset snapshot: {args.dataset_pt2}")

    fp32 = common.apply_precision(args)
    common.check_manifest(args.package, fp32)
    common.check_eager_meta(args.dataset_pt2, fp32)
    print(f"loading compiled package {args.package} ...")
    runner = torch._inductor.aoti_load_package(str(args.package))

    scenes = common.load_dataset_scenes(args.dataset_pt2, args.device)
    references = common.load_dataset_eager_depth(args.dataset_pt2, args.device)
    totals = {key: 0.0 for key in ("count", "abs_error", "sq_error", "max_abs", "max_rel")}
    evaluated = 0

    for (scene, inputs), (ref_scene, reference) in zip(scenes, references, strict=True):
        if scene != ref_scene:
            raise RuntimeError(f"snapshot scene mismatch: {scene!r} != {ref_scene!r}")
        if reference is None:
            print(f"[{scene}] skipped: no stored eager reference "
                  f"(snapshot predates depth_eager -- re-run export_dataset.py)")
            continue

        with torch.no_grad():
            prediction = runner(*inputs)
        prediction = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        if prediction.ndim == 4 and prediction.shape[0] == 1:
            prediction = prediction[0]

        try:
            sums = diff_sums(prediction, reference)
        except ValueError as error:
            raise SystemExit(f"{scene}: {error}") from error
        for key in ("count", "abs_error", "sq_error"):
            totals[key] += sums[key]
        for key in ("max_abs", "max_rel"):
            totals[key] = max(totals[key], sums[key])
        evaluated += 1
        print(f"[{scene}] {format_metrics(summarize(sums))}  pixels={int(sums['count'])}")

    if evaluated == 0:
        raise SystemExit("dataset snapshot contains no scenes with a stored eager reference")

    print(f"\noverall ({evaluated} scene(s), {int(totals['count'])} pixels)")
    print(format_metrics(summarize(totals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
