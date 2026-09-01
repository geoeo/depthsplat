#!/usr/bin/env python3
"""Export a dataset cfg_dict plus every staged scene's raw tensors to a .pt2.

    python scripts/export_dataset.py --dataset realm_1_with_depth \
        --offset 70 --count 3 --output outputs/dataset_cfg.pt2

Scene selection matches scripts/realm_triplet_depth.py; the triplet is generated
from custom/<dataset>/ on each run. The resulting .pt2 is self-contained --
load it with depth_export_common.load_dataset_scenes()/load_dataset_cfg(), and
pass it to aot_compile_depth.py/run_depth_so.py via --dataset-pt2.
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
    parser.add_argument("--output", type=Path,
                        default=common.REPO_ROOT / "outputs" / "dataset_cfg.pt2",
                        help="destination .pt2 holding cfg_dict and raw scene tensors")
    args = parser.parse_args()
    common.validate_scene_selection(args)

    common.prepare_scenes(args)
    cfg_dict = common.build_config(args.data_root)
    saved = common.export_dataset_cfg(cfg_dict, args.output, num_views=args.num_views)
    size_mb = saved.stat().st_size / 2**20
    print(f"saved dataset cfg + raw scene tensors to {saved} ({size_mb:.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
