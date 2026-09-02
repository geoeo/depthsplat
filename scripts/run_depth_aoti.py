#!/usr/bin/env python3
"""Run a compiled AOTInductor depth package over realm triplets.

Takes the same scene-selection arguments as scripts/realm_triplet_depth.py --
the triplet scenes are generated from custom/<dataset>/ on each run, so this
does not depend on a previously populated dataset directory:

    python scripts/run_depth_aoti.py \
        --package outputs/aoti/fp32/depth_predictor_fp32.pt2 \
        --dataset realm_1_with_depth --offset 70 --count 3

    python scripts/run_depth_aoti.py --package ... \
        --dataset realm_1_with_depth --offset 70 --indices '[0,1,2,9,10,11]' \
        --compare-eager --benchmark

--fp32 defaults to on, matching the default of aot_compile_depth.py. It must
agree with how the package was built: the precision is not recorded inside the
package, and a mismatch is silent -- the depths are simply wrong by metres. The
build directory's build_info.json is checked at startup to catch exactly that,
along with a torch version that no longer matches the one that compiled it.

One caveat on --compare-eager: importing depth_export_common sets
AOTI_SAFE_INVERSE, so the eager model built here for comparison uses the same
export-safe inverse the package was compiled from (see src/export_compat.py).
The reported diff is therefore package-vs-inductor, which is what you want when
checking that a build is sound. It is *not* the gap against the depths
src/main.py produces, which is larger because the pipeline still uses
torch.inverse; that gap is quoted in aot_compile_depth.py.

Step 2 of 2. This is the Python stand-in for the C++ loader: it drives the same
.pt2 through torch._inductor.aoti_load_package(), the counterpart of C++'s
torch::inductor::AOTIModelPackageLoader.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `src` import: sets XFORMERS_DISABLED / TYPECHECK_DISABLED.
import depth_export_common as common  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.misc.image_io import save_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    parser.add_argument("--package", type=Path, required=True,
                        help="compiled .pt2 package to run (from aot_compile_depth.py)")
    common.add_precision_arg(parser)
    parser.add_argument("--compare-eager", action="store_true",
                        help="also build the eager model and diff against it. NOTE: "
                             "that model is built in this process, so it runs with "
                             "AOTI_SAFE_INVERSE set too -- the diff measures the "
                             "package against the same arithmetic it was compiled "
                             "from, not against what src/main.py computes. See "
                             "src/export_compat.py.")
    parser.add_argument("--benchmark", action="store_true", help="time the package")
    parser.add_argument("--iters", type=int, default=8, help="benchmark iterations")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="write images/<scene>/depth/{idx}.npy here, matching "
                             "realm_triplet_depth.py's directory layout")
    parser.add_argument("--save-cam-images", action="store_true",
                        help="also write each view's input camera image to --output-dir")
    args = parser.parse_args()
    common.validate_scene_selection(args)

    if not args.package.is_file():
        raise SystemExit(
            f"no such package: {args.package} (build it with aot_compile_depth.py)"
        )
    if args.package.suffix == ".so":
        raise SystemExit(
            f"{args.package} is a bare .so from a pre-2.6 build. Recompile with "
            f"aot_compile_depth.py to get a .pt2 package."
        )

    fp32 = common.apply_precision(args)
    common.check_manifest(args.package, fp32)

    print(f"staging scenes from custom/{args.dataset} ...")
    rows = common.prepare_scenes(args)
    cfg_dict = common.build_config_for_args(args)

    print(f"loading {args.package} ...")
    # No device argument: the package is compiled for one device and carries it.
    # --device still decides where the input tensors are placed, and must agree.
    runner = torch._inductor.aoti_load_package(str(args.package))

    model = None
    if args.compare_eager:
        print("building eager model for comparison ...")
        model = common.build_model(cfg_dict, args.device, args.num_views)

    if args.output_dir:
        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    worst = 0.0
    count = 0
    for scene, inputs, depth_gt in common.iter_inputs_with_gt(cfg_dict, args.device, args.num_views):
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
            depth_dir = args.output_dir / "images" / scene / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            for v in range(depth.shape[1]):
                path = depth_dir / f"{v:0>6}.npy"
                np.save(path, depth[0, v].cpu().numpy())
                print(f"  wrote {path}")
            if model is not None:
                for v in range(reference.shape[1]):
                    eager_path = depth_dir / f"{v:0>6}_eager.npy"
                    np.save(eager_path, reference[0, v].cpu().numpy())
                    print(f"  wrote {eager_path}")
            if args.save_cam_images:
                images = inputs[0]
                for v in range(images.shape[1]):
                    image_path = depth_dir / f"{v:0>6}_cam.png"
                    save_image(images[0, v], image_path)
                    print(f"  wrote {image_path}")
            if depth_gt is not None:
                for v in range(depth_gt.shape[0]):
                    gt_path = depth_dir / f"{v:0>6}_gt.npy"
                    np.save(gt_path, depth_gt[v].cpu().numpy())
                    print(f"  wrote {gt_path}")

    print(f"\nran {count} scene(s) from {len(rows)} staged triplet(s)")
    if model is not None:
        print(f"worst maxabs across scenes: {worst:.3e} m")
        if worst > 1e-2:
            print("NOTE: a metre-scale gap usually means --fp32 disagrees with the "
                  "build, or the package was built with --fp32 off (see "
                  "aot_compile_depth.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
