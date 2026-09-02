#!/usr/bin/env python3
"""Export the depth predictor to a torch.export ExportedProgram.

    python scripts/export_depth_model.py --dataset realm_1_with_depth \
        --offset 70 --count 3 --output outputs/depth_predictor.pt2

Scene selection matches scripts/realm_triplet_depth.py; the triplet is generated
from custom/<dataset>/ on each run and its first scene becomes the tracing
example.

The result is a serialised graph, not a state dict -- load it with
`torch.export.load`, never `torch.load`. It is device- and shape-specialised:
V, H and W are baked in.

Steps:
    export_dataset.py      -> .pt2  (dataset cfg + raw tensors) [Optional]
    export_depth_model.py  -> .pt2  (serialised graph) [Optional]
    aot_compile_depth.py   -> .pt2  (AOTInductor package: compiled .so + kernels)
    run_depth_aoti.py      -> runs the package

Three different things wear the .pt2 extension here, and they are not
interchangeable:

    torch.save()                        -> read with torch.load()
      (export_dataset.py's snapshot: a pickled dict of cfg + tensors)
    torch.export.save()                 -> read with torch.export.load()
      (this script: an ExportedProgram, still interpreted at runtime)
    aoti_compile_and_package()          -> read with aoti_load_package()
      (aot_compile_depth.py: native code, no Python interpreter in the loop)

This script is optional -- aot_compile_depth.py re-exports the model itself
rather than consuming the graph written here. Its value is inspection: the node
count, and the banned-op scan below.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `src` import: sets XFORMERS_DISABLED / TYPECHECK_DISABLED.
import depth_export_common as common  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--output", type=Path,
                        default=common.REPO_ROOT / "outputs" / "depth_predictor.pt2",
                        help="destination .pt2 (torch.export convention; a .pt "
                             "state dict is a different thing)")
    parser.add_argument("--strict", choices=["true", "false"], default="true",
                        help="dynamo-based tracing (true, torch's own default) or "
                             "non-strict. Both work; strict is the stronger check.")
    common.add_precision_arg(parser)
    parser.add_argument("--skip-check", action="store_true",
                        help="skip the eager-vs-exported comparison")
    args = parser.parse_args()

    common.apply_precision(args)

    print("building model ...")
    model, inputs = common.build_model_and_inputs(args)

    with torch.no_grad():
        reference = model(*inputs).clone()
    print(f"  eager depth {tuple(reference.shape)} "
          f"range=[{reference.min():.3f}, {reference.max():.3f}] m")

    strict = args.strict == "true"
    print(f"exporting (strict={strict}) ...")
    with torch.no_grad():
        exported = torch.export.export(model, inputs, strict=strict)
    print(f"  {len(list(exported.graph.nodes))} graph nodes")

    targets = [str(n.target) for n in exported.graph.nodes]
    for banned in ("cudnn", "flags", "CheckpointFunction", "memory_efficient"):
        if any(banned in t for t in targets):
            print(f"  WARNING: '{banned}' present in the graph")

    if not args.skip_check:
        with torch.no_grad():
            got = exported.module()(*inputs)
        diff = (got - reference).abs()
        rel = (diff / reference.abs().clamp(min=1e-6)).max()
        print(f"  exported vs eager: maxabs={diff.max():.3e} m maxrel={rel:.3e} "
              f"bitexact={torch.equal(got, reference)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(args.output))
    size_mb = args.output.stat().st_size / 2**20
    print(f"saved {args.output} ({size_mb:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
