"""In-process depth-inference API.

Mirrors the test path of `src/main.py` but is callable from Python instead of
requiring a `python -m src.main` subprocess with Hydra CLI overrides.
"""

import os
from pathlib import Path
import warnings

import torch
from colorama import Fore
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.precision import describe_precision, set_fp32_precision
from src.typecheck import typecheck_hook

# Configure beartype and jaxtyping. Set TYPECHECK_DISABLED in the environment
# before importing this module to skip the hook (required for torch.export()).
with typecheck_hook():
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_PRETRAINED_DEPTH = "pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth"


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def compose_config(overrides: list[str]) -> DictConfig:
    """Build the same DictConfig that `@hydra.main` produces for `src/main.py`."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name="main", overrides=overrides)


def depth_inference_overrides(
    dataset_root: Path | str,
    output_dir: Path | str,
    image_shape: tuple[int, int],
    near: float,
    far: float,
    scene_selection: str | None = None,
    pretrained_depth: str = DEFAULT_PRETRAINED_DEPTH,
    vit_type: str = "vitb",
) -> list[str]:
    """Overrides for depth-only inference over a `custom_images` root.

    Uses the `all` view sampler so every frame of a scene becomes a context view,
    which removes the need for an evaluation-index JSON file.
    """
    h, w = image_shape
    return [
        "+experiment=re10k",
        "dataset=custom_images",
        "dataset/view_sampler=all",
        f"dataset.roots=[{dataset_root}]",
        f"dataset.scene_selection={'null' if scene_selection is None else scene_selection}",
        "dataset.test_chunk_interval=1",
        f"dataset.image_shape=[{h},{w}]",
        f"dataset.near={near}",
        f"dataset.far={far}",
        "mode=test",
        "model.encoder.num_scales=2",
        "model.encoder.upsample_factor=4",
        "model.encoder.lowest_feature_resolution=8",
        f"model.encoder.monodepth_vit_type={vit_type}",
        "model.encoder.train_depth_only=true",
        "train.forward_depth_only=true",
        f"checkpointing.pretrained_depth={pretrained_depth}",
        "test.compute_scores=false",
        "test.save_depth=true",
        "test.save_depth_concat_img=true",
        "test.save_depth_npy=true",
        f"output_dir={output_dir}",
        "wandb.mode=disabled",
    ]


def run_depth_inference(cfg_dict: DictConfig, full_fp32: bool = True) -> Path:
    """Run depth-only testing in this process. Returns the output directory.

    `full_fp32` defaults to True so the depth maps written here can serve as a
    numerical reference for an AOTInductor build, which agrees with eager to
    ~1e-03 m in fp32 but only to ~6 m under TF32. Pass False for the older,
    ~19% faster TF32 behaviour. See src/precision.py.
    """
    warnings.filterwarnings("ignore")
    set_fp32_precision(full_fp32)
    print(cyan(f"Precision: {'full fp32' if full_fp32 else 'TF32'} "
               f"({describe_precision()})"))

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
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    state = torch.load(cfg.checkpointing.pretrained_depth, map_location="cpu")["model"]
    model_wrapper.encoder.depth_predictor.load_state_dict(state, strict=True)
    print(cyan(f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"))

    trainer.test(model_wrapper, datamodule=data_module, ckpt_path=None)
    return output_dir

