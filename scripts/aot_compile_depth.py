#!/usr/bin/env python3
"""Compile the depth predictor to a native .so with AOTInductor.

    python scripts/aot_compile_depth.py \
        --dataset realm_1_with_depth --offset 70 --count 3
    python scripts/aot_compile_depth.py --fp32 off \
        --dataset realm_1_with_depth --offset 70 --count 3 \
        --output-dir /opt/models/depth

Scene selection matches scripts/realm_triplet_depth.py; the triplet is generated
from custom/<dataset>/ on each run and its first scene becomes the tracing
example.

Each build gets its own directory, because the .so is NOT self-contained: it
loads ~108 generated Triton kernels from separate .cubin files at runtime.

Those paths are ABSOLUTE and fixed at compile time -- they are NOT resolved
relative to the .so. Copying the .so somewhere else keeps working only for as
long as the original build directory still exists; on a machine where it does
not, inference fails with an opaque `run_func_ ... API call failed`. Note it
fails on the first *inference*, not on load, so a smoke test that only
constructs the runner will pass.

So: deploy the directory to the very path it was compiled for, or compile with
--output-dir set to where it will live on the target.

--fp32 is the load-bearing choice. It is baked into the generated kernels, and
the runtime must be set to match (run_depth_so.py --fp32); the setting is
recorded in build_info.json so a mismatch can be caught. Measured on a 3-view
480x640 scene, RTX 3060:

    --fp32 on  (default)  515 ms   agrees with fp32 eager to ~8.7e-04 m
    --fp32 off (TF32)     391 ms   differs from eager by ~6.5 m max / 0.48 m mean

The TF32 gap is not a broken graph -- pinned to fp32 the same build agrees with
eager to fp32 rounding. Inductor simply selects different kernels than eager,
and the cost-volume regression amplifies that into metres. fp32 is the default
because it is what reproduces the Python pipeline; choose `off` only when the
~25% speedup is worth metre-scale disagreement.

Step 2 of 3. Takes a few minutes; torch 2.4 uses torch._export.aot_compile.
"""

import argparse
import re
import shutil
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
    common.add_precision_arg(parser)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="build directory, created if absent (default: "
                             "outputs/aoti/<tf32|fp32>/). Holds the .so and its .cubin "
                             "files; deploy it as a unit. Absolute paths to this "
                             "directory are compiled into the .so.")
    parser.add_argument("--clean", action="store_true",
                        help="wipe the build directory first (stale .cubin files from "
                             "an earlier build are otherwise left behind)")
    parser.add_argument("--skip-check", action="store_true",
                        help="skip loading the .so and comparing against eager")
    args = parser.parse_args()

    fp32 = args.fp32 == "on"
    variant = "fp32" if fp32 else "tf32"
    build_dir = (args.output_dir or common.REPO_ROOT / "outputs" / "aoti" / variant).resolve()
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"cleaned {build_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"depth_predictor_{variant}.so"

    common.apply_precision(args)

    print("building model ...")
    model, inputs = common.build_model_and_inputs(args)

    with torch.no_grad():
        reference = model(*inputs).clone()
    print(f"  eager depth {tuple(reference.shape)} "
          f"range=[{reference.min():.3f}, {reference.max():.3f}] m")

    print(f"compiling -> {build_dir}/ (several minutes) ...")
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
        if not fp32 and diff.max() > 1e-2:
            print("  (expected under --fp32 off; rebuild with --fp32 on to compare "
                  "against eager at fp32 rounding)")

    common.write_manifest(build_dir, fp32, output.name)
    audit(Path(so_path), build_dir)

    print(f"\nbuild directory: {build_dir}")
    print(f"run it with: python scripts/run_depth_so.py --so {so_path} "
          f"--fp32 {args.fp32}")
    return 0


def audit(so_path: Path, build_dir: Path) -> None:
    """Report the .so's runtime dependency on its .cubin files.

    The .so calls cuModuleLoad on absolute paths fixed at compile time, so a build
    is only portable if it stays at the path it was compiled for.
    """
    cubins = sorted(build_dir.glob("*.cubin"))
    mib = lambda fs: sum(f.stat().st_size for f in fs) / 2**20
    print(f"  {len(cubins)} .cubin kernels alongside the .so ({mib(cubins):.1f} MiB)")

    # .o/.cpp are build intermediates -- the generated source is handy for the C++
    # integration, but neither is needed at runtime.
    leftovers = sorted(build_dir.glob("*.o")) + sorted(build_dir.glob("*.cpp"))
    runtime = mib([so_path]) + mib(cubins)
    print(f"  deployable: .so + .cubin = {runtime:.0f} MiB")
    if leftovers:
        print(f"  build intermediates (not needed at runtime): "
              f"{len(leftovers)} files, {mib(leftovers):.0f} MiB")

    refs = {Path(m.decode()) for m in
            re.findall(rb"/[ -~]{1,240}?\.cubin", so_path.read_bytes())}
    outside = sorted(r for r in refs if r.parent != build_dir)
    missing = sorted(r for r in refs if not r.is_file())
    print(f"  {len(refs)} cubin paths baked into the .so")
    if outside:
        print(f"  WARNING: {len(outside)} reference a directory other than the build "
              f"dir, e.g. {outside[0]}")
    if missing:
        print(f"  WARNING: {len(missing)} baked paths do not exist, e.g. {missing[0]}")
    if not outside and not missing:
        print(f"  all baked paths resolve inside the build dir")
        print(f"  NOTE: paths are absolute, not relative to the .so -- this build "
              f"only runs where {build_dir} exists")


if __name__ == "__main__":
    raise SystemExit(main())
