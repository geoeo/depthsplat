#!/usr/bin/env python3
"""Compile the depth predictor to a native .so with AOTInductor.

    python scripts/aot_compile_depth.py --tf32 on   --output outputs/depth_tf32.so
    python scripts/aot_compile_depth.py --tf32 off  --output outputs/depth_fp32.so

--tf32 is the load-bearing choice here. It is baked into the generated kernels,
and the runtime must be set to match (run_depth_so.py --tf32). Measured on a
3-view 480x640 scene, RTX 3060:

    --tf32 on    391 ms   differs from eager by ~6.5 m max / 0.48 m mean
    --tf32 off   515 ms   differs from eager by ~8.7e-04 m max

The large TF32 gap is not a broken graph -- with TF32 pinned off the same build
agrees with eager to fp32 rounding. Inductor simply picks different kernels than
eager, and the cost-volume regression amplifies that into metres. Choose `off`
if the C++ output is validated numerically against Python; `on` for the ~32%
speedup if metre-scale disagreement is acceptable.

Step 2 of 3. Takes a few minutes; torch 2.4 uses torch._export.aot_compile.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `src` import: sets XFORMERS_DISABLED / TYPECHECK_DISABLED.
import depth_export_common as common  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--tf32", choices=["on", "off"], required=True,
                        help="REQUIRED, no default: baked into the kernels and must "
                             "match the runtime. See the note above.")
    parser.add_argument("--output", type=Path, default=None,
                        help="destination .so (default: outputs/depth_predictor_<tf32>.so)")
    parser.add_argument("--skip-check", action="store_true",
                        help="skip loading the .so and comparing against eager")
    args = parser.parse_args()

    tf32 = args.tf32 == "on"
    output = args.output or (common.REPO_ROOT / "outputs" /
                             f"depth_predictor_{'tf32' if tf32 else 'fp32'}.so")

    common.set_tf32(tf32)
    print(f"precision: {common.describe_precision()}")

    print("building model ...")
    model, inputs = common.build_model_and_inputs(args)

    with torch.no_grad():
        reference = model(*inputs).clone()
    print(f"  eager depth {tuple(reference.shape)} "
          f"range=[{reference.min():.3f}, {reference.max():.3f}] m")

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"compiling -> {output} (several minutes) ...")
    started = time.time()
    with torch.no_grad():
        so_path = torch._export.aot_compile(
            model, inputs, options={"aot_inductor.output_path": str(output)},
        )
    print(f"  compiled in {time.time() - started:.0f}s "
          f"({Path(so_path).stat().st_size / 2**20:.0f} MiB)")

    if not args.skip_check:
        runner = torch._export.aot_load(str(so_path), args.device)
        with torch.no_grad():
            got = runner(*inputs)
        got = got[0] if isinstance(got, (list, tuple)) else got
        diff = (got - reference).abs()
        rel = diff / reference.abs().clamp(min=1e-6)
        print(f"  .so vs eager: maxabs={diff.max():.3e} m maxrel={rel.max():.3e} "
              f"mean={diff.mean():.3e} m")
        if tf32 and diff.max() > 1e-2:
            print("  (expected under --tf32 on; rebuild with --tf32 off to compare "
                  "against eager at fp32 rounding)")

    print(f"\nsaved {so_path}")
    print(f"run it with: python scripts/run_depth_so.py --so {so_path} "
          f"--tf32 {args.tf32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
