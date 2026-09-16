#!/usr/bin/env python3
"""Compile the depth predictor to a self-contained AOTInductor .pt2 package.

    python scripts/aot_compile_depth.py \
        --dataset realm_1_with_depth --offset 70 --count 3
    python scripts/aot_compile_depth.py --fp32 off \
        --dataset realm_1_with_depth --offset 70 --count 3 \
        --output-dir /opt/models/depth

Scene selection matches scripts/realm_triplet_depth.py; the triplet is generated
from custom/<dataset>/ on each run and its first scene becomes the tracing
example.

This uses the stable two-step export workflow, as of torch 2.6:

    ep      = torch.export.export(model, inputs)
    package = torch._inductor.aoti_compile_and_package(ep, package_path=...)

superseding torch._export.aot_compile(). That call still exists in 2.6 but
prints a deprecation banner, and what it produced was a bare .so that loaded its
generated Triton kernels from separate .cubin files by ABSOLUTE path, fixed at
compile time. A build was therefore only usable from Python at the exact
directory it was compiled into -- and it failed on the first *inference* rather
than at load, so a smoke test that merely constructed the runner would pass.
C++ could relocate such a build only by passing a cubin_dir to the runner.

The .pt2 package removes all of that: the compiled .so and every .cubin live
inside one zip archive, which the loader unpacks to a temporary directory. The
package is a single self-contained file -- copy it anywhere.

    Python: torch._inductor.aoti_load_package(path)
    C++:    torch::inductor::AOTIModelPackageLoader loader(path);
            loader.run(inputs);

A package is still locked to the torch that built it, so it must be recompiled
after a torch upgrade. build_info.json records the version and
run_depth_aoti.py reports a mismatch at startup.

--fp32 is the load-bearing choice. It is baked into the generated kernels, and
the runtime must be set to match (run_depth_aoti.py --fp32); the setting is
recorded in build_info.json so a mismatch can be caught.

                         package    eager   package vs eager, same precision
    --fp32 on (default)   160 ms   177 ms   6.8e-03 m max, 2.1e-04 m mean
    --fp32 off (TF32)     123 ms   155 ms   2.68 m max, 3.2e-02 m mean

Worst case over the four scenes: 9.2e-03 m for fp32, 2.68 m for TF32.

The TF32 gap is not a broken graph -- pinned to fp32 the same package agrees
with eager to fp32 rounding. Inductor simply selects different kernels than
eager, and the cost-volume regression amplifies that into metres. fp32 is the
default because it is what reproduces the Python pipeline; choose `off` only
when the ~23% saving is worth metre-scale disagreement.

Against what src/main.py actually computes, an fp32 package lands within
5.3e-03 m (mean 2.1e-04 m). That differs from the package-vs-eager figure above
because the pipeline still calls torch.inverse while the exported graph uses the
substitution in src/export_compat.py; that substitution accounts for 5.5e-03 m
max / 3.3e-04 m mean on its own.

Re-measured 2026-09-02 on torch 2.6.0+cu126 / RTX 4070 Ti SUPER, over the
four 3-view 480x640 scenes in outputs/dataset_cfg.pt2. The figures this
replaces were taken on torch 2.4.0+cu124 / RTX 3060; both the toolchain and
the GPU changed, so old and new are not directly comparable.

Step 1 of 2. Takes a few minutes.
"""

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `src` import: sets XFORMERS_DISABLED / TYPECHECK_DISABLED.
import depth_export_common as common  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    common.add_model_args(parser)
    common.add_precision_arg(parser)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="build directory, created if absent (default: "
                             "outputs/aoti/<tf32|fp32>/). Holds the .pt2 package and "
                             "its build_info.json; the package alone is enough to run "
                             "inference and can be copied anywhere.")
    parser.add_argument("--clean", action="store_true",
                        help="wipe the build directory first (also clears the loose "
                             ".so/.cubin litter left by pre-2.6 builds)")
    parser.add_argument("--skip-check", action="store_true",
                        help="skip loading the package and comparing against eager")
    args = parser.parse_args()
    common.validate_scene_selection(args)

    fp32 = args.fp32 == "on"
    variant = "fp32" if fp32 else "tf32"
    build_dir = (args.output_dir or common.REPO_ROOT / "outputs" / "aoti" / variant).resolve()
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"cleaned {build_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / f"depth_predictor_{variant}.pt2"

    common.apply_precision(args)

    print("building model ...")
    model, inputs = common.build_model_and_inputs(args)

    with torch.no_grad():
        reference = model(*inputs).clone()
    print(f"  eager depth {tuple(reference.shape)} "
          f"range=[{reference.min():.3f}, {reference.max():.3f}] m")

    print("exporting the graph ...")
    with torch.no_grad():
        exported = torch.export.export(model, inputs)
    print(f"  {len(list(exported.graph.nodes))} graph nodes")

    print(f"compiling -> {output} (several minutes) ...")
    started = time.time()
    with torch.no_grad():
        package = Path(torch._inductor.aoti_compile_and_package(
            exported, package_path=str(output),
        ))
    print(f"  compiled in {time.time() - started:.0f}s "
          f"({package.stat().st_size / 2**20:.0f} MiB)")

    if not args.skip_check:
        runner = torch._inductor.aoti_load_package(str(package))
        with torch.no_grad():
            got = runner(*inputs)
        got = got[0] if isinstance(got, (list, tuple)) else got
        diff = (got - reference).abs()
        rel = diff / reference.abs().clamp(min=1e-6)
        print(f"  package vs eager: maxabs={diff.max():.3e} m maxrel={rel.max():.3e} "
              f"mean={diff.mean():.3e} m")
        if not fp32 and diff.max() > 1e-2:
            print("  (expected under --fp32 off; rebuild with --fp32 on to compare "
                  "against eager at fp32 rounding)")

    common.write_manifest(build_dir, fp32, output.name)
    audit(package)

    print(f"\nbuild directory: {build_dir}")
    print(f"run it with: python scripts/run_depth_aoti.py --package {package} "
          f"--fp32 {args.fp32}")
    return 0


def audit(package: Path) -> None:
    """Report what the .pt2 package contains, by file type.

    Everything the runtime needs is inside the archive -- the compiled .so and the
    generated .cubin kernels -- and nothing is referenced by an absolute path, so
    the package is relocatable. This just makes the contents visible.
    """
    with zipfile.ZipFile(package) as archive:
        entries = [e for e in archive.infolist() if not e.is_dir()]

    mib = lambda items: sum(i.file_size for i in items) / 2**20
    by_ext: dict[str, list[zipfile.ZipInfo]] = {}
    for entry in entries:
        by_ext.setdefault(Path(entry.filename).suffix or "(none)", []).append(entry)

    print(f"  {len(entries)} entries, {mib(entries):.0f} MiB uncompressed:")
    for ext, items in sorted(by_ext.items(), key=lambda kv: -mib(kv[1])):
        print(f"    {len(items):>4} {ext:<8} {mib(items):>8.1f} MiB")
    if not by_ext.get(".so"):
        print("  WARNING: no .so inside the package")
    print(f"  self-contained: the .cubin kernels ship inside the archive, so the "
          f"package needs no sibling files and can be copied anywhere")


if __name__ == "__main__":
    raise SystemExit(main())
