"""Shared pieces for the depth-model export pipeline.

IMPORTANT: importing this module sets `XFORMERS_DISABLED`, `TYPECHECK_DISABLED`
and `AOTI_SAFE_INVERSE` before anything under `src` is imported. All three are
read at *import* time and all three are needed for the export/compile path to
produce a working artifact:

  * xformers' `memory_efficient_attention` is an opaque custom op with no ATen
    lowering. The fallback is `scaled_dot_product_attention`, which exports fine.
  * the jaxtyping/beartype import hook rewrites every `forward`, so the tracer
    would capture the beartype wrapper instead of the real function.
  * `aten.linalg_inv_ex` has no c-shim in torch 2.6, so AOTInductor routes
    `torch.inverse` through the proxy executor, which mis-deserializes one of its
    arguments and hands back an undefined tensor. That one fails at *inference*
    rather than at compile time; see `src/export_compat.py`.

So `import depth_export_common` must come before any `src` import.
"""

import os

os.environ["XFORMERS_DISABLED"] = "1"
os.environ["TYPECHECK_DISABLED"] = "1"
os.environ["AOTI_SAFE_INVERSE"] = "1"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.depth_inference import (  # noqa: E402
    DEFAULT_PRETRAINED_DEPTH,
    compose_config,
    depth_inference_overrides,
)
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

    A flat positional tensor signature is what the C++ loader wants, and it keeps
    the exported graph's input spec trivial. Everything non-tensor
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

    fp32 is the default because it is what reproduces the Python pipeline. An fp32
    package agrees with fp32 eager to 6.8e-03 m max (2.1e-04 m mean), and with the
    depths src/main.py computes to 5.3e-03 m. A TF32 package disagrees with TF32
    eager by 2.68 m max (3.2e-02 m mean), and TF32 vs fp32 within eager alone is
    already 2.87 m max. TF32 saves roughly 23% latency and costs that agreement.
    See scripts/aot_compile_depth.py for the full table and how it was measured.
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


def manifest_path(package_path: Path) -> Path:
    return Path(package_path).parent / "build_info.json"


def write_manifest(build_dir: Path, fp32: bool, package_name: str) -> None:
    """Record how a build was compiled, next to the .pt2 package."""
    (build_dir / "build_info.json").write_text(json.dumps({
        "fp32": fp32,
        "package": package_name,
        "torch": torch.__version__,
        "created": datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")


def check_manifest(package_path: Path, fp32: bool) -> None:
    """Warn if the runtime disagrees with how the package was built.

    Two mismatches are worth catching before inference rather than after:

      * precision. Not recorded inside the package, and a mismatch fails
        silently -- the depths are simply wrong by metres.
      * torch version. An AOTInductor build is locked to the torch that produced
        it, both its C++ ABI and its runtime API, so a package compiled under a
        different version either fails to load or is subtly wrong. See
        .github/agents/pytorch-cxx11-abi.md.
    """
    path = manifest_path(package_path)
    if not path.is_file():
        print(f"  no build_info.json beside the package; cannot verify that --fp32 "
              f"{'on' if fp32 else 'off'} matches how it was built")
        return
    info = json.loads(path.read_text())
    built_fp32 = info["fp32"]
    if built_fp32 == fp32:
        print(f"  build_info.json: compiled with fp32={'on' if built_fp32 else 'off'} -- matches")
    else:
        print(f"  *** MISMATCH: package was compiled with fp32="
              f"{'on' if built_fp32 else 'off'} but running with fp32="
              f"{'on' if fp32 else 'off'}. Depths will be wrong by metres. ***")

    built_torch = info.get("torch")
    if built_torch and built_torch != torch.__version__:
        print(f"  *** MISMATCH: package was compiled with torch {built_torch}, "
              f"running torch {torch.__version__}. An AOTInductor build is not "
              f"portable across torch versions -- recompile it with "
              f"aot_compile_depth.py. ***")


def set_tf32(enabled: bool) -> None:
    """Delegates to src.precision, so these scripts and the pipeline cannot drift.

    This model amplifies TF32 heavily: eager TF32 vs eager full-fp32 differs by
    ~2.87 m max / 5.8e-02 m mean on a 3-view scene, and an AOTInductor package
    compiled under TF32 disagrees with TF32 eager by ~2.68 m max / 3.2e-02 m mean
    -- not because the graph is wrong, but because inductor selects different
    kernels. Pinned to full fp32 the same package agrees to 6.8e-03 m, at roughly
    30% more latency.

    Whatever is chosen at compile time must also be set by the runtime.
    """
    set_fp32_precision(full_fp32=not enabled)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Scene-selection flags, mirroring scripts/realm_triplet_depth.py."""
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--indices",
                        help="Flat index list relative to --offset, e.g. '[0,1,2,3,4,5]'")
    source.add_argument("--count", type=int,
                        help="Use consecutive indices range(count); must be a multiple of 3")
    parser.add_argument("--dataset-pt2", type=Path, default=None,
                        help="Optional snapshot of the cfg_dict produced by the export script; "
                             "if set, it takes precedence over --dataset/--count/--offset")
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


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Depth-model architecture and trained-checkpoint selection."""
    parser.add_argument("--backbone", choices=["vits", "vitb"], default="vitb",
                    help="DINOv2 monodepth backbone (default: vitb). ViT-L export "
                        "is not supported.")
    parser.add_argument("--checkpoint", type=Path,
                    default=Path(DEFAULT_PRETRAINED_DEPTH),
                    help="matching trained depth-predictor checkpoint (default: "
                        f"{DEFAULT_PRETRAINED_DEPTH}).")


def validate_scene_selection(args):
    """Require some scene selection, unless a dataset snapshot already fixes it."""
    if getattr(args, "dataset_pt2", None) is not None:
        return
    if args.count is None and args.indices is None:
        raise SystemExit("either --dataset-pt2 or one of --count/--indices is required")


def prepare_scenes(args):
    """Build the triplet scenes from custom/<dataset>/, as realm_triplet_depth.py does."""
    if getattr(args, "dataset_pt2", None) is not None:
        print(f"  using dataset snapshot {args.dataset_pt2}; ignoring --dataset/--count/--offset")
        return []

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


def build_config(data_root: Path, backbone: str | None = None,
                 checkpoint: Path | None = None):
    dataset_cfg = yaml.safe_load(DATASET_CFG_PATH.read_text())
    model_overrides = {}
    if backbone is not None:
        model_overrides["vit_type"] = backbone
    if checkpoint is not None:
        model_overrides["pretrained_depth"] = str(checkpoint)
    return compose_config(
        depth_inference_overrides(
            dataset_root=data_root.resolve().relative_to(REPO_ROOT),
            output_dir="outputs/export_tmp",
            image_shape=tuple(dataset_cfg["image_shape"]),
            near=dataset_cfg["near"],
            far=dataset_cfg["far"],
            **model_overrides,
        )
    )


def export_dataset_cfg(cfg_dict: DictConfig | dict, output_path: str | Path,
                        num_views: int | None = None, device: str = "cuda",
                        fp32: bool = True) -> Path:
    """Serialize cfg_dict, every staged scene's raw tensors, and the eager depth
    the Python model produces for them, to a .pt2 snapshot.

    Embedding the tensors (image, intrinsics, extrinsics, near/far) makes the
    snapshot self-contained: load_dataset_scenes()/load_dataset_eager_depth() do
    not re-read dataset.roots off disk; only the cfg_dict's non-tensor settings
    (image_shape, near, far, ...) are still path-shaped metadata.

    `depth_eager` is the reference a compiled package is scored against: the
    eager model's own output, not a measurement. It replaces the external depth
    reference field earlier snapshots carried. Those were the custom/<dataset>/dense .npy
    maps, which are unscaled OpenREALM stereo -- their sibling .txt says
    "Scaling (Not Georeferenced)" -- and are not multi-view consistent (r~0.03
    reprojected between neighbouring views, against r~0.87 for the model). They
    cannot tell a sound build from a broken one, so they are no longer embedded.

    The eager output depends on `fp32`: TF32 eager and full-fp32 eager differ by
    metres on this model, so the precision is recorded alongside and checked by
    check_eager_meta() before any comparison.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = OmegaConf.to_container(cfg_dict, resolve=True)
    scenes = []
    if num_views is not None:
        model = build_model(cfg_dict, device, num_views)
        for scene, batch in _iter_batches(cfg_dict, num_views):
            ctx = batch["context"]
            with torch.no_grad():
                depth_eager = model(
                    ctx["image"].to(device),
                    ctx["intrinsics"].to(device),
                    ctx["extrinsics"].to(device),
                    (1.0 / ctx["far"]).to(device),
                    (1.0 / ctx["near"]).to(device),
                )
            # [B, V, H, W] -> [V, H, W]; B is 1 here, and dropping it matches the
            # shape evaluate_depth_aoti.py reduces a package's output to.
            if depth_eager.ndim == 4:
                depth_eager = depth_eager[0]
            depth_eager = depth_eager.detach().cpu()
            scenes.append({
                "scene": scene,
                "images": ctx["image"],
                "intrinsics": ctx["intrinsics"],
                "extrinsics": ctx["extrinsics"],
                "min_depth": (1.0 / ctx["far"]),
                "max_depth": (1.0 / ctx["near"]),
                "depth_eager": depth_eager,
            })
            print(f"  {scene}: embedding eager depth {tuple(depth_eager.shape)} "
                  f"range=[{depth_eager.min():.3f}, {depth_eager.max():.3f}] m")
    torch.save({
        "cfg_dict": payload,
        "scenes": scenes,
        "eager": {
            "fp32": fp32,
            "torch": str(torch.__version__),
            "num_views": num_views,
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    }, str(path))
    return path


def _load_reference_depth(depth_reference_paths) -> torch.Tensor | None:
    """Stack per-view reference depth .npy files into [V, H, W], or None if absent.

    `depth_reference_paths` is a list of V entries, each collated to a batch-size-1
    list by the dataloader ("" when a view has no reference); mirrors how
    model_wrapper.py reads the same field.
    """
    if depth_reference_paths is None:
        return None
    paths = [
        path[0] if isinstance(path, (list, tuple)) else path
        for path in depth_reference_paths
    ]
    if not paths or not all(p and Path(p).is_file() for p in paths):
        return None
    return torch.from_numpy(np.stack([np.load(p) for p in paths], axis=0)).float()


SNAPSHOT_SAFE_GLOBALS: tuple[type, ...] = ()
"""Types a snapshot payload may hold beyond tensors and plain containers.

Deliberately empty, and it should stay that way.

A snapshot has to remain readable by libtorch's IValue unpickler, because
`.github/agents/cpp-inference.md` points a C++ input pipeline at this file as the
reference for building correct input tensors. That unpickler resolves only a
fixed set of pickle GLOBALs -- `torch._utils._rebuild_tensor_v2`, the storage
types, `collections.OrderedDict`. Anything else makes the *whole* file
unreadable from C++, not just the offending key, because the payload
deserializes into one IValue.

`torch.__version__` is exactly that trap: it is a `TorchVersion`, not a `str`,
and storing it directly emits a fourth GLOBAL that libtorch cannot construct.
Hence the `str()` at the write site in export_dataset_cfg().

Python alone would accept such a type, via
`torch.serialization.safe_globals(...)`, which is what this tuple feeds. Adding
to it trades away C++ readability -- so do not, unless that has been given up
deliberately.
"""


def _load_snapshot(path: str | Path) -> dict:
    """Read a .pt2 snapshot written by export_dataset_cfg().

    The single entry point for reading these files, so the contract above is
    stated once instead of being implied by four separate `torch.load` calls.

    torch 2.6's `weights_only=True` default is left in place. Against an empty
    allowlist it is not merely a security setting: it fails on precisely the
    types libtorch cannot read, so every Python-side load of a snapshot doubles
    as a check that the C++ reader could still open it.
    """
    if SNAPSHOT_SAFE_GLOBALS:
        with torch.serialization.safe_globals(list(SNAPSHOT_SAFE_GLOBALS)):
            return torch.load(str(path), map_location="cpu")
    return torch.load(str(path), map_location="cpu")


def load_dataset_cfg(path: str | Path) -> DictConfig:
    """Load a cfg_dict snapshot produced by export_dataset_cfg()."""
    payload = _load_snapshot(path)
    if isinstance(payload, dict) and "cfg_dict" in payload:
        payload = payload["cfg_dict"]
    return OmegaConf.create(payload)


def load_dataset_scenes(path: str | Path, device: str = "cpu"):
    """Load the embedded (scene, flat input tuple) pairs saved by export_dataset_cfg()."""
    payload = _load_snapshot(path)
    scenes = payload["scenes"] if isinstance(payload, dict) else []
    for row in scenes:
        yield row["scene"], (
            row["images"].to(device),
            row["intrinsics"].to(device),
            row["extrinsics"].to(device),
            row["min_depth"].to(device),
            row["max_depth"].to(device),
        )


def load_dataset_eager_depth(path: str | Path, device: str = "cpu"):
    """Yield (scene, depth_eager) for every embedded scene; None when absent.

    depth_eager is [V, H, W]: what the eager Python model produced for that scene
    at export time, and what a compiled package is diffed against. Snapshots
    written before this field existed carry an external depth reference instead
    and yield None.
    """
    payload = _load_snapshot(path)
    scenes = payload["scenes"] if isinstance(payload, dict) else []
    for row in scenes:
        depth = row.get("depth_eager")
        yield row["scene"], depth.to(device) if depth is not None else None


def load_dataset_eager_meta(path: str | Path) -> dict:
    """The precision and torch version the snapshot's depth_eager was produced at."""
    payload = _load_snapshot(path)
    return payload.get("eager", {}) if isinstance(payload, dict) else {}


def check_eager_meta(dataset_path: str | Path, fp32: bool) -> None:
    """Warn when the snapshot's eager reference was produced at another precision.

    The same silent failure check_manifest() guards for the package: nothing in
    the stored tensor says which precision produced it, and TF32 eager differs
    from full-fp32 eager by metres on this model -- a mismatch here reads as a
    broken compile when it is only a rounding-mode difference.
    """
    meta = load_dataset_eager_meta(dataset_path)
    if not meta:
        print("  snapshot carries no eager metadata; cannot verify the reference "
              "precision (it predates depth_eager -- re-export it)")
        return
    if meta.get("fp32") == fp32:
        print(f"  snapshot eager reference: fp32={'on' if fp32 else 'off'} -- matches")
    else:
        print(f"  *** MISMATCH: snapshot's eager reference was produced with fp32="
              f"{'on' if meta.get('fp32') else 'off'} but running with fp32="
              f"{'on' if fp32 else 'off'}. The diff will be precision noise in "
              f"metres, not a compile error. ***")
    made_with = meta.get("torch")
    if made_with and made_with != torch.__version__:
        print(f"  note: eager reference came from torch {made_with}, "
              f"running torch {torch.__version__}")


def build_config_for_args(args):
    """Return the active cfg_dict, preferring a dataset snapshot when supplied."""
    dataset_pt2 = getattr(args, "dataset_pt2", None)
    backbone = getattr(args, "backbone", None)
    checkpoint = getattr(args, "checkpoint", None)
    if dataset_pt2 is not None:
        if not dataset_pt2.is_file():
            raise SystemExit(f"no such dataset snapshot: {dataset_pt2}")
        print(f"loading dataset cfg from {dataset_pt2}")
        cfg_dict = load_dataset_cfg(dataset_pt2)
        if backbone is not None:
            cfg_dict.model.encoder.monodepth_vit_type = backbone
        if checkpoint is not None:
            cfg_dict.checkpointing.pretrained_depth = str(checkpoint)
        return cfg_dict
    return build_config(args.data_root, backbone, checkpoint)


def build_model(cfg_dict, device: str, num_views: int) -> DepthExport:
    """Build the depth predictor, load pretrained weights, freeze V, wrap it."""
    cfg = load_typed_root_config(cfg_dict)
    if cfg.model.encoder.monodepth_vit_type == "vitl":
        raise SystemExit(
            "ViT-L export is not supported: no complete ViT-L DepthSplat depth "
            "checkpoint is available, and the training initializer is not exportable."
        )
    set_cfg(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)
    print(f"  monodepth backbone: {cfg.model.encoder.monodepth_vit_type}")

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
    for scene, batch in _iter_batches(cfg_dict, num_views):
        ctx = batch["context"]
        yield scene, (
            ctx["image"].to(device),
            ctx["intrinsics"].to(device),
            ctx["extrinsics"].to(device),
            (1.0 / ctx["far"]).to(device),
            (1.0 / ctx["near"]).to(device),
        )


def iter_inputs_with_reference(cfg_dict, device: str, num_views: int):
    """Yield (scene_name, flat input tuple, depth_reference) for every staged scene.

    depth_reference is a [V, H, W] tensor on `device`, or None when unavailable.
    """
    for scene, batch in _iter_batches(cfg_dict, num_views):
        ctx = batch["context"]
        depth_reference = _load_reference_depth(ctx.get("depth_reference_path"))
        yield scene, (
            ctx["image"].to(device),
            ctx["intrinsics"].to(device),
            ctx["extrinsics"].to(device),
            (1.0 / ctx["far"]).to(device),
            (1.0 / ctx["near"]).to(device),
        ), depth_reference.to(device) if depth_reference is not None else None


def _iter_batches(cfg_dict, num_views: int):
    """Yield (scene_name, batch) for every staged scene, enforcing the view count V."""
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
        yield scene, batch


def first_inputs(cfg_dict, device: str, num_views: int):
    """The first staged scene, used as the tracing example."""
    scene, inputs = next(iter_inputs(cfg_dict, device, num_views))
    print(f"  tracing example from scene {scene}")
    print("  input shapes: " + ", ".join(str(tuple(t.shape)) for t in inputs))
    return inputs


def build_model_and_inputs(args):
    """Stage scenes, build the model, and return it with one example input tuple."""
    validate_scene_selection(args)
    prepare_scenes(args)
    cfg_dict = build_config_for_args(args)
    model = build_model(cfg_dict, args.device, args.num_views)
    return model, first_inputs(cfg_dict, args.device, args.num_views)
