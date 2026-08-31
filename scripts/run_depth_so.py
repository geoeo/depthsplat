#!/usr/bin/env python3
"""Run a compiled AOTInductor depth .so over realm triplets.

Takes the same scene-selection arguments as scripts/realm_triplet_depth.py --
the triplet scenes are generated from custom/<dataset>/ on each run, so this
does not depend on a previously populated dataset directory:

    python scripts/run_depth_so.py --so outputs/aoti/fp32/depth_predictor_fp32.so \
        --dataset realm_1_with_depth --offset 70 --count 3

    python scripts/run_depth_so.py --so ... \
        --dataset realm_1_with_depth --offset 70 --indices '[0,1,2,9,10,11]' \
        --compare-eager --benchmark

--fp32 defaults to on, matching the default of aot_compile_depth.py. It must
agree with how the .so was built: the precision is not recorded inside the .so,
and a mismatch is silent -- the depths are simply wrong by metres. The build
directory's build_info.json is checked at startup to catch exactly that.

Step 3 of 3. This is the Python stand-in for the C++ loader: it drives the same
.so through torch._export.aot_load.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `src` import: sets XFORMERS_DISABLED / TYPECHECK_DISABLED.
import depth_export_common as common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--so", type=Path, required=True, help="compiled .so to run")
    common.add_precision_arg(parser)
    parser.add_argument("--compare-eager", action="store_true",
                        help="also build the eager model and diff against it")
    parser.add_argument("--benchmark", action="store_true", help="time the .so")
    parser.add_argument("--iters", type=int, default=8, help="benchmark iterations")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="write <scene>.npy depth maps here")
    args = parser.parse_args()
    common.validate_scene_selection(args)

    if not args.so.is_file():
        raise SystemExit(f"no such .so: {args.so} (build it with aot_compile_depth.py)")

    fp32 = common.apply_precision(args)
    common.check_manifest(args.so, fp32)

    print(f"staging scenes from custom/{args.dataset} ...")
    rows = common.prepare_scenes(args)
    cfg_dict = common.build_config_for_args(args)

    print(f"loading {args.so} ...")
    runner = torch._export.aot_load(str(args.so), args.device)

    model = None
    if args.compare_eager:
        print("building eager model for comparison ...")
        model = common.build_model(cfg_dict, args.device, args.num_views)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    worst = 0.0
    count = 0
    for scene, inputs in common.iter_inputs(cfg_dict, args.device, args.num_views):
        count += 1
        with torch.no_grad():
            depth = runner(*inputs)
        depth = depth[0] if isinstance(depth, (list, tuple)) else depth
        print(f"\n[{scene}]")
        print(f"  depth {tuple(depth.shape)} on {depth.device} "
              f"range=[{depth.min():.3f}, {depth.max():.3f}] m mean={depth.mean():.3f} m")

        if model is not None:
            with torch.no_grad():
                reference = model(*inputs)
            diff = (depth - reference).abs()
            rel = (diff / reference.abs().clamp(min=1e-6)).max()
            worst = max(worst, diff.max().item())
            print(f"  vs eager: maxabs={diff.max():.3e} m maxrel={rel:.3e} "
                  f"mean={diff.mean():.3e} m bitexact={torch.equal(depth, reference)}")

        if args.benchmark:
            with torch.no_grad():
                runner(*inputs)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(args.iters):
                    runner(*inputs)
                torch.cuda.synchronize()
            per_call = (time.perf_counter() - started) / args.iters * 1000
            print(f"  {per_call:.1f} ms/call over {args.iters} iters "
                  f"({1000 / per_call:.2f} calls/s)")

        if args.output_dir:
            path = args.output_dir / f"{scene}.npy"
            np.save(path, depth.cpu().numpy())
            print(f"  wrote {path}")

    print(f"\nran {count} scene(s) from {len(rows)} staged triplet(s)")
    if model is not None:
        print(f"worst maxabs across scenes: {worst:.3e} m")
        if worst > 1e-2:
            print("NOTE: a metre-scale gap usually means --fp32 disagrees with the "
                  "build, or the .so was built with --fp32 off (see aot_compile_depth.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
