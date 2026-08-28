"""Shared pieces for the depth-model export pipeline.

IMPORTANT: importing this module sets `XFORMERS_DISABLED` and `TYPECHECK_DISABLED`
before anything under `src` is imported. Both are read at *import* time and both
must be off for `torch.export()` to work:

  * xformers' `memory_efficient_attention` is an opaque custom op with no ATen
    lowering. The fallback is `scaled_dot_product_attention`, which exports fine.
  * the jaxtyping/beartype import hook rewrites every `forward`, so the tracer
    would capture the beartype wrapper instead of the real function.

So `import depth_export_common` must come before any `src` import.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"
os.environ["TYPECHECK_DISABLED"] = "1"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.depth_inference import compose_config, depth_inference_overrides  # noqa: E402
from src.config import load_typed_root_config  # noqa: E402
from src.global_cfg import set_cfg  # noqa: E402
from src.misc.step_tracker import StepTracker  # noqa: E402
from src.dataset.data_module import DataModule  # noqa: E402
from src.model.encoder import get_encoder  # noqa: E402
from src.model.encoder.unimatch.mv_unimatch import set_num_views  # noqa: E402
from src.precision import describe_precision, set_fp32_precision  # noqa: E402

# Scenes are generated on demand from custom/<dataset>/ into this staging root,
# exactly as scripts/realm_triplet_depth.py does. Nothing here depends on a
# previously populated dataset directory.
DEFAULT_STAGING_ROOT = REPO_ROOT / "datasets" / "custom_images_triplets"
DATASET_CFG_PATH = REPO_ROOT / "config" / "dataset" / "custom_images.yaml"
DEFAULT_ATTN_SPLITS = [2]


class DepthExport(torch.nn.Module):
    """The deployment signature: tensors in positionally, one depth tensor out.

    AOTInductor on torch 2.4 does not accept keyword arguments, and a flat tensor
    signature is what the C++ loader wants in any case. Everything non-tensor
    (`attn_splits_list`, `nn_matrix`) is baked in as a constant.

        depth = model(images, intrinsics, extrinsics, min_depth, max_depth)

        images      [B, V, 3, H, W]
        intrinsics  [B, V, 3, 3]      normalised by image width/height
        extrinsics  [B, V, 4, 4]      camera-to-world, first view at identity
        min_depth   [B, V]            1 / far
        max_depth   [B, V]            1 / near
        -> depth    [B, V, H, W]      metres
    """

    def __init__(self, core: torch.nn.Module, attn_splits_list: list[int]):
        super().__init__()
        self.core = core
        self.attn_splits_list = attn_splits_list

    def forward(self, images, intrinsics, extrinsics, min_depth, max_depth):
        out = self.core(
            images,
            attn_splits_list=self.attn_splits_list,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            min_depth=min_depth,
            max_depth=max_depth,
            nn_matrix=None,
        )
        return out["depth_preds"][-1]


def add_precision_arg(parser: argparse.ArgumentParser) -> None:
    """`--fp32 on|off`, defaulting to full fp32.

    fp32 is the default because it is what reproduces the Python pipeline. Measured
    against the depth maps the pipeline writes, an fp32 build lands within 1.30 m
    (mean 0.105 m) and a TF32 build within 6.48 m (mean 0.484 m); once the Python
    reference is also generated in fp32, the fp32 build matches it to 8.7e-04 m.
    TF32 buys roughly 25% latency and costs that agreement.
    """
    parser.add_argument("--fp32", choices=["on", "off"], default="on",
                        help="on (default): full fp32, matches the Python pipeline "
                             "most closely. off: allow TF32, ~25%% faster but "
                             "disagrees with Python by metres.")


def apply_precision(args) -> bool:
    """Apply --fp32 and report. Returns True when running in full fp32."""
    fp32 = args.fp32 == "on"
    set_tf32(not fp32)
    print(f"precision: fp32={args.fp32} ({describe_precision()})")
    return fp32


def manifest_path(so_path: Path) -> Path:
    return Path(so_path).parent / "build_info.json"


def write_manifest(build_dir: Path, fp32: bool, so_name: str) -> None:
    """Record the precision a build was compiled with, next to the .so."""
    (build_dir / "build_info.json").write_text(json.dumps({
        "fp32": fp32,
        "so": so_name,
        "torch": torch.__version__,
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")


def check_manifest(so_path: Path, fp32: bool) -> None:
    """Warn if the runtime precision disagrees with what the .so was built with.

    The setting is not recorded inside the .so itself, and a mismatch fails
    silently -- the depths are simply wrong by metres.
    """
    path = manifest_path(so_path)
    if not path.is_file():
        print(f"  no build_info.json beside the .so; cannot verify that --fp32 "
              f"{'on' if fp32 else 'off'} matches how it was built")
        return
    built_fp32 = json.loads(path.read_text())["fp32"]
    if built_fp32 == fp32:
        print(f"  build_info.json: compiled with fp32={'on' if built_fp32 else 'off'} -- matches")
    else:
        print(f"  *** MISMATCH: .so was compiled with fp32="
              f"{'on' if built_fp32 else 'off'} but running with fp32="
              f"{'on' if fp32 else 'off'}. Depths will be wrong by metres. ***")


def set_tf32(enabled: bool) -> None:
    """Delegates to src.precision, so these scripts and the pipeline cannot drift.

    This model amplifies TF32 heavily: eager TF32 vs eager full-fp32 differs by
    ~1.30 m max / 0.105 m mean on a 3-view scene, and an AOTInductor build compiled
    under TF32 disagrees with eager by ~6.5 m max / 0.48 m mean -- not because the
    graph is wrong, but because inductor selects different kernels. Pinned to full
    fp32 the same build agrees to 8.7e-04 m, at roughly 25% more latency.

    Whatever is chosen at compile time must also be set by the runtime.
    """
    set_fp32_precision(full_fp32=not enabled)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Scene-selection flags, mirroring scripts/realm_triplet_depth.py."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--indices",
                        help="Flat index list relative to --offset, e.g. '[0,1,2,3,4,5]'")
    source.add_argument("--count", type=int,
                        help="Use consecutive indices range(count); must be a multiple of 3")
    parser.add_argument("--dataset", default="realm_1",
                        choices=["realm_1", "realm_2", "realm_1_with_depth"])
    parser.add_argument("--offset", type=int, default=0,
                        help="Index of the first frame; indices are relative to it")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Divide translations by this factor")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_STAGING_ROOT,
                        help="Staging root for the generated scenes (rebuilt each run)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-views", type=int, default=3,
                        help="V baked into the graph. Static: the UNet attention "
                             "factorises the batch dim by it.")


def prepare_scenes(args):
    """Build the triplet scenes from custom/<dataset>/, as realm_triplet_depth.py does."""
    from realm_triplet_depth import parse_indices, preprocess

    if args.count is not None:
        if args.count <= 0 or args.count % 3 != 0:
            raise SystemExit(f"--count must be a positive multiple of 3, got {args.count}")
        indices = list(range(args.count))
    else:
        indices = parse_indices(args.indices)

    rows, warnings = preprocess(args.dataset, indices, args.data_root, args.scale, args.offset)
    for warning in warnings:
        print(f"  WARNING: {warning}")
    return rows


def build_config(data_root: Path):
    dataset_cfg = yaml.safe_load(DATASET_CFG_PATH.read_text())
    return compose_config(
        depth_inference_overrides(
            dataset_root=data_root.resolve().relative_to(REPO_ROOT),
            output_dir="outputs/export_tmp",
            image_shape=tuple(dataset_cfg["image_shape"]),
            near=dataset_cfg["near"],
            far=dataset_cfg["far"],
        )
    )


def build_model(cfg_dict, device: str, num_views: int) -> DepthExport:
    """Build the depth predictor, load pretrained weights, freeze V, wrap it."""
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)

    ckpt = cfg.checkpointing.pretrained_depth
    state = torch.load(ckpt, map_location="cpu")["model"]
    encoder.depth_predictor.load_state_dict(state, strict=True)
    print(f"  loaded pretrained depth: {ckpt}")

    core = encoder.depth_predictor.to(device).eval()

    # Freeze the view count *before* tracing. MultiViewUniMatch.forward would
    # otherwise mutate attention.n_frames from inside the exported region; the
    # guard there makes it a no-op once this matches.
    set_num_views(core.regressor, num_views=num_views)
    core._configured_num_views = num_views
    print(f"  num_views frozen at {num_views}")

    return DepthExport(core, DEFAULT_ATTN_SPLITS).to(device).eval()


def iter_inputs(cfg_dict, device: str, num_views: int):
    """Yield (scene_name, flat input tuple) for every staged scene."""
    cfg = load_typed_root_config(cfg_dict)
    loader = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0).test_dataloader()
    for batch in loader:
        ctx = batch["context"]
        v = ctx["image"].shape[1]
        if v != num_views:
            raise SystemExit(
                f"scene has {v} views but --num-views is {num_views}; they must match, "
                f"because V is static in the exported graph"
            )
        scene = batch["scene"][0] if isinstance(batch.get("scene"), list) else batch.get("scene")
        yield scene, (
            ctx["image"].to(device),
            ctx["intrinsics"].to(device),
            ctx["extrinsics"].to(device),
            (1.0 / ctx["far"]).to(device),
            (1.0 / ctx["near"]).to(device),
        )


def first_inputs(cfg_dict, device: str, num_views: int):
    """The first staged scene, used as the tracing example."""
    scene, inputs = next(iter_inputs(cfg_dict, device, num_views))
    print(f"  tracing example from scene {scene}")
    print("  input shapes: " + ", ".join(str(tuple(t.shape)) for t in inputs))
    return inputs


def build_model_and_inputs(args):
    """Stage scenes, build the model, and return it with one example input tuple."""
    prepare_scenes(args)
    cfg_dict = build_config(args.data_root)
    model = build_model(cfg_dict, args.device, args.num_views)
    return model, first_inputs(cfg_dict, args.device, args.num_views)
