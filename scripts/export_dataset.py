#!/usr/bin/env python3
"""Export a dataset cfg_dict, every staged scene's raw tensors, and the eager
reference depth for those tensors, to a .pt2.

    python scripts/export_dataset.py --dataset realm_1_with_depth \
        --offset 70 --count 3 --output outputs/dataset_cfg.pt2

Scene selection matches scripts/realm_triplet_depth.py; the triplet is generated
from custom/<dataset>/ on each run. The resulting .pt2 is self-contained --
load it with depth_export_common.load_dataset_scenes()/load_dataset_cfg(), and
pass it to aot_compile_depth.py/run_depth_aoti.py via --dataset-pt2.

Each scene also carries `depth_eager`: the depth the eager Python model produces
for that scene, stored as [V, H, W]. That is the reference evaluate_depth_aoti.py
scores a compiled package against. It replaces the external depth reference,
which held the custom/<dataset>/dense .npy maps -- unscaled OpenREALM stereo that is not
multi-view consistent and so cannot distinguish a sound build from a broken one.

Because the model is run here, this script needs the checkpoint and a GPU, and
--fp32 must match what the package under test was compiled with. The precision
is recorded in the snapshot and checked before any comparison.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must precede any `src` import: sets XFORMERS_DISABLED / TYPECHECK_DISABLED.
import depth_export_common as common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    common.add_model_args(parser)
    common.add_precision_arg(parser)
    parser.add_argument("--output", type=Path,
                        default=common.REPO_ROOT / "outputs" / "dataset_cfg.pt2",
                        help="destination .pt2 holding cfg_dict and raw scene tensors")
    args = parser.parse_args()
    common.validate_scene_selection(args)

    # Before the model is built: the eager reference is only valid at the
    # precision it was produced under, and that is what gets recorded.
    fp32 = common.apply_precision(args)

    common.prepare_scenes(args)
    cfg_dict = common.build_config_for_args(args)
    saved = common.export_dataset_cfg(cfg_dict, args.output, num_views=args.num_views,
                                      device=args.device, fp32=fp32)
    size_mb = saved.stat().st_size / 2**20
    print(f"saved dataset cfg + raw scene tensors + eager reference to {saved} "
          f"({size_mb:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
