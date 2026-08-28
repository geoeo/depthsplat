#!/usr/bin/env python3
"""Export the depth predictor to a torch.export ExportedProgram.

    python scripts/export_depth_model.py --output outputs/depth_predictor.pt2

The result is a serialised graph, not a state dict -- load it with
`torch.export.load`, never `torch.load`. It is device- and shape-specialised:
V, H and W are baked in.

Step 2 of 3:
    export_depth_model.py    -> .pt2   (graph)
    aot_compile_depth.py     -> .so    (compiled kernels)
    run_depth_so.py          -> runs the .so
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
                        help="dynamo-based tracing (true) or non-strict. Both work; "
                             "strict is the stronger check.")
    parser.add_argument("--tf32", choices=["on", "off"], default="on",
                        help="precision used for the numerical check below. Does not "
                             "change the graph -- only the eager/exported comparison.")
    parser.add_argument("--skip-check", action="store_true",
                        help="skip the eager-vs-exported comparison")
    args = parser.parse_args()

    common.set_tf32(args.tf32 == "on")
    print(f"precision: {common.describe_precision()}")

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
