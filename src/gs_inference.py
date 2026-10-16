"""In-process Gaussian-splatting inference (test mode) on custom_images scenes.

Mirrors `src/depth_inference.py` for the full DepthSplat model: compose the Hydra
config with `+experiment=realm` (so image_shape, near and far come from
config/dataset/custom_images.yaml and apply to the whole scene), load a
`depthsplat-gs-*` checkpoint into the whole ModelWrapper, and run Trainer.test
with colour renders, scores, Gaussians and rendered depth enabled.

The precision switch is the reason this exists next to `python -m src.main`,
which always runs TF32: evaluations should be reproducible in full fp32.
"""

import os
import warnings
from pathlib import Path

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.depth_inference import (
    DataModule,
    LocalLogger,
    ModelWrapper,
    StepTracker,
    compose_config,
    cyan,
    get_decoder,
    get_encoder,
    get_losses,
    load_typed_root_config,
    set_cfg,
)
from src.precision import describe_precision, set_fp32_precision

DEFAULT_PRETRAINED_GS = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"


def gs_inference_overrides(
    dataset_root: Path | str,
    output_dir: Path | str,
    index_path: Path | str,
    scene_selection: str | None = None,
    pretrained_model: str = DEFAULT_PRETRAINED_GS,
    pretrained_depth: str | None = None,
    vit_type: str = "vitb",
    num_scales: int = 2,
    upsample_factor: int = 4,
    lowest_feature_resolution: int = 8,
    gaussian_scale_max: float = 0.1,
    normalize_scene_scale: bool = True,
    scene_scale_mode: str = "nearfar",
    scene_scale_target_depth: float = 3.0,
    scene_scale_baseline: float = 1.0,
    save_gaussian: bool = True,
    save_rendered_depth: bool = True,
    compute_scores: bool = True,
) -> list[str]:
    """Hydra overrides for a GS test run over a custom_images root with an evaluation index.

    No image_shape / near / far here on purpose: `+experiment=realm` leaves them to
    config/dataset/custom_images.yaml.

    `normalize_scene_scale` is on by default: the GS checkpoints were trained at room
    scale and collapse to the far plane on metric aerial scenes (verified 2026-10-16),
    so the encoder rescales the context internally and returns metric outputs.
    `gaussian_scale_max` is then in normalised units (sqrt(near*far) = scene_scale_target_depth).

    `pretrained_depth` builds a hybrid: after the GS checkpoint, the depth predictor is
    overwritten with a depth-only checkpoint (same architecture; `src/main.py` loads the
    two in this order too). Combine with normalize_scene_scale=False, because the
    depth-only checkpoints are metric.
    """
    return [
        "+experiment=realm",
        f"dataset.roots=[{dataset_root}]",
        f"dataset.scene_selection={'null' if scene_selection is None else scene_selection}",
        "dataset.test_chunk_interval=1",
        "dataset/view_sampler=realm",
        f"dataset.view_sampler.index_path={index_path}",
        "mode=test",
        f"model.encoder.num_scales={num_scales}",
        f"model.encoder.upsample_factor={upsample_factor}",
        f"model.encoder.lowest_feature_resolution={lowest_feature_resolution}",
        f"model.encoder.monodepth_vit_type={vit_type}",
        f"model.encoder.gaussian_adapter.gaussian_scale_max={gaussian_scale_max}",
        f"model.encoder.normalize_scene_scale={str(normalize_scene_scale).lower()}",
        f"model.encoder.scene_scale_mode={scene_scale_mode}",
        f"model.encoder.scene_scale_target_depth={scene_scale_target_depth}",
        f"model.encoder.scene_scale_baseline={scene_scale_baseline}",
        f"checkpointing.pretrained_model={pretrained_model}",
        f"checkpointing.pretrained_depth={'null' if pretrained_depth is None else pretrained_depth}",
        f"test.compute_scores={str(compute_scores).lower()}",
        "test.save_image=true",
        "test.save_gt_image=true",
        f"test.save_gaussian={str(save_gaussian).lower()}",
        "test.save_depth=true",
        "test.save_depth_npy=true",
        f"test.save_rendered_depth={str(save_rendered_depth).lower()}",
        "data_loader.test.num_workers=0",
        "wandb.mode=disabled",
        f"output_dir={output_dir}",
    ]


def run_gs_inference(cfg_dict: DictConfig, full_fp32: bool = True) -> Path:
    """Run full-model testing in this process. Returns the output directory."""
    warnings.filterwarnings("ignore")
    set_fp32_precision(full_fp32)
    print(cyan(f"Precision: {'full fp32' if full_fp32 else 'TF32'} ({describe_precision()})"))

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    output_dir = Path(cfg_dict.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    step_tracker = StepTracker()
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=LocalLogger(),
        devices=1,
        strategy="auto",
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    torch.manual_seed(cfg_dict.seed)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=None,
    )
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker, global_rank=trainer.global_rank)

    state = torch.load(cfg.checkpointing.pretrained_model, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model_wrapper.load_state_dict(state, strict=not cfg.checkpointing.no_strict_load)
    print(cyan(f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"))
    if cfg.checkpointing.pretrained_depth is not None:
        depth_state = torch.load(cfg.checkpointing.pretrained_depth, map_location="cpu")["model"]
        model_wrapper.encoder.depth_predictor.load_state_dict(depth_state, strict=True)
        print(cyan(f"Loaded pretrained depth (hybrid): {cfg.checkpointing.pretrained_depth}"))

    trainer.test(model_wrapper, datamodule=data_module, ckpt_path=None)
    return output_dir
