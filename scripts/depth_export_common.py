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
import sys
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

DEFAULT_DATA_ROOT = REPO_ROOT / "datasets" / "custom_images_triplets"
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


def set_tf32(enabled: bool) -> None:
    """Pin the float32 matmul path.

    This model amplifies TF32 heavily: eager TF32 vs eager full-fp32 differs by
    ~1.30 m max / 0.105 m mean on a 3-view scene, and an AOTInductor build compiled
    under TF32 disagrees with eager by ~6.5 m max / 0.48 m mean -- not because the
    graph is wrong, but because inductor selects different kernels. Pinned to full
    fp32 the same build agrees to 8.7e-04 m, at roughly 32% more latency.

    Whatever is chosen at compile time must also be set by the runtime.
    """
    torch.set_float32_matmul_precision("high" if enabled else "highest")
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def describe_precision() -> str:
    return (
        f"matmul_precision={torch.get_float32_matmul_precision()} "
        f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}"
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="custom_images root holding a test/<scene>/ tree "
                             "(populate it with scripts/realm_triplet_depth.py)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-views", type=int, default=3,
                        help="V baked into the graph. Static: the UNet attention "
                             "factorises the batch dim by it.")


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


def example_inputs(cfg_dict, device: str, num_views: int):
    """One real batch from the dataset, as the flat positional tuple."""
    cfg = load_typed_root_config(cfg_dict)
    loader = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0).test_dataloader()
    batch = next(iter(loader))
    ctx = batch["context"]
    v = ctx["image"].shape[1]
    if v != num_views:
        raise SystemExit(
            f"scene has {v} views but --num-views is {num_views}; they must match, "
            f"because V is static in the exported graph"
        )
    inputs = (
        ctx["image"].to(device),
        ctx["intrinsics"].to(device),
        ctx["extrinsics"].to(device),
        (1.0 / ctx["far"]).to(device),
        (1.0 / ctx["near"]).to(device),
    )
    print("  example inputs: " + ", ".join(str(tuple(t.shape)) for t in inputs))
    return inputs


def build_model_and_inputs(args):
    cfg_dict = build_config(args.data_root)
    model = build_model(cfg_dict, args.device, args.num_views)
    inputs = example_inputs(cfg_dict, args.device, args.num_views)
    return model, inputs
