#!/usr/bin/env python3
"""Run a compiled AOTInductor depth .so.

    python scripts/run_depth_so.py --so outputs/depth_predictor_fp32.so --tf32 off
    python scripts/run_depth_so.py --so ... --tf32 off --compare-eager --benchmark

--tf32 must match what the .so was compiled with (aot_compile_depth.py --tf32).
It is not recorded in the .so, so nothing will warn you if it disagrees -- the
depths will just be quietly wrong by metres.

Step 3 of 3. This is the Python stand-in for the C++ loader: it exercises the
same .so through torch._export.aot_load.
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
    parser.add_argument("--tf32", choices=["on", "off"], required=True,
                        help="REQUIRED: must match the .so's compile-time setting")
    parser.add_argument("--compare-eager", action="store_true",
                        help="also build the eager model and diff against it")
    parser.add_argument("--benchmark", action="store_true", help="time the .so")
    parser.add_argument("--iters", type=int, default=8, help="benchmark iterations")
    parser.add_argument("--save-npy", type=Path, default=None,
                        help="write the depth maps to this .npy")
    args = parser.parse_args()

    if not args.so.is_file():
        raise SystemExit(f"no such .so: {args.so} (build it with aot_compile_depth.py)")

    common.set_tf32(args.tf32 == "on")
    print(f"precision: {common.describe_precision()}")

    # Inputs come from the same dataset path the model was compiled against; the
    # eager model is only built when a comparison is asked for.
    cfg_dict = common.build_config(args.data_root)
    inputs = common.example_inputs(cfg_dict, args.device, args.num_views)

    print(f"loading {args.so} ...")
    runner = torch._export.aot_load(str(args.so), args.device)

    with torch.no_grad():
        depth = runner(*inputs)
    depth = depth[0] if isinstance(depth, (list, tuple)) else depth
    print(f"  depth {tuple(depth.shape)} on {depth.device} "
          f"range=[{depth.min():.3f}, {depth.max():.3f}] m "
          f"mean={depth.mean():.3f} m")

    if args.compare_eager:
        print("building eager model for comparison ...")
        model = common.build_model(cfg_dict, args.device, args.num_views)
        with torch.no_grad():
            reference = model(*inputs)
        diff = (depth - reference).abs()
        rel = diff / reference.abs().clamp(min=1e-6)
        print(f"  .so vs eager: maxabs={diff.max():.3e} m maxrel={rel.max():.3e} "
              f"mean={diff.mean():.3e} m bitexact={torch.equal(depth, reference)}")
        if diff.max() > 1e-2:
            print("  NOTE: a metre-scale gap usually means --tf32 disagrees with the "
                  "build, or the .so was built with --tf32 on (see aot_compile_depth.py)")

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

    if args.save_npy:
        args.save_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy, depth.cpu().numpy())
        print(f"  wrote {args.save_npy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
