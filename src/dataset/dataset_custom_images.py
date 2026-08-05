import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetCustomImagesCfg(DatasetCfgCommon):
    name: Literal["custom_images"]
    roots: list[Path]
    metadata_file: str
    intrinsics_are_normalized: bool
    extrinsics_are_c2w: bool
    augment: bool
    train_times_per_scene: int
    test_times_per_scene: int
    test_chunk_interval: int
    shuffle_val: bool
    skip_bad_shape: bool
    scene_selection: str | None
    near: float
    far: float
    max_fov: float


class DatasetCustomImages(IterableDataset):
    cfg: DatasetCustomImagesCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    scene_dirs: list[Path]

    def __init__(
        self,
        cfg: DatasetCustomImagesCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.scene_dirs = self._collect_scene_dirs()

    def _collect_scene_dirs(self) -> list[Path]:
        scene_dirs: list[Path] = []
        for root in self.cfg.roots:
            root = Path(root)  # Ensure root is a Path object
            stage_root = root / self.stage
            search_root = stage_root if stage_root.is_dir() else root
            if not search_root.is_dir():
                continue

            # If root itself contains the metadata file, treat it as a single scene.
            if (search_root / self.cfg.metadata_file).is_file():
                scene_dirs.append(search_root)
                continue

            for path in sorted(search_root.iterdir()):
                if path.is_dir() and (path / self.cfg.metadata_file).is_file():
                    # Filter by scene_selection if specified
                    if self.cfg.scene_selection is None or path.name == self.cfg.scene_selection:
                        scene_dirs.append(path)
        return scene_dirs

    def _load_scene(self, scene_dir: Path) -> tuple[str, Tensor, Tensor, Tensor]:
        metadata_path = scene_dir / self.cfg.metadata_file
        with metadata_path.open("r") as f:
            metadata = json.load(f)

        frames = metadata["frames"]
        scene_name = metadata.get("scene", scene_dir.name)

        images = []
        intrinsics = []
        extrinsics = []
        for frame in frames:
            image_path = scene_dir / frame["image"]
            image = Image.open(image_path).convert("RGB")
            images.append(self.to_tensor(image))

            k = torch.tensor(frame["intrinsics"], dtype=torch.float32)
            if not self.cfg.intrinsics_are_normalized:
                _, h, w = images[-1].shape
                k = k.clone()
                k[0] /= float(w)
                k[1] /= float(h)
            intrinsics.append(k)

            e = torch.tensor(frame["extrinsics"], dtype=torch.float32)
            if not self.cfg.extrinsics_are_c2w:
                e = e.inverse()
            extrinsics.append(e)

        return (
            scene_name,
            torch.stack(images),
            torch.stack(extrinsics),
            torch.stack(intrinsics),
        )

    def _get_bound(self, value: float, num_views: int) -> Tensor:
        return torch.full((num_views,), float(value), dtype=torch.float32)

    def _shuffle(self, lst: list) -> list:
        if not lst:
            return lst
        indices = torch.randperm(len(lst))
        return [lst[i] for i in indices]

    def __iter__(self):
        scene_dirs = self.scene_dirs
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train",)):
            scene_dirs = self._shuffle(scene_dirs)

        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            scene_dirs = [
                scene_dir
                for i, scene_dir in enumerate(scene_dirs)
                if i % worker_info.num_workers == worker_info.id
            ]

        times_per_scene = (
            self.cfg.test_times_per_scene
            if self.stage == "test"
            else self.cfg.train_times_per_scene
        )

        for scene_dir in scene_dirs:
            scene, images, extrinsics, intrinsics = self._load_scene(scene_dir)

            if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                continue

            if self.stage == "test":
                # Systematically cover all frames: chunk frame indices into non-overlapping
                # windows of num_context_views so every frame gets a depth prediction.
                num_views = extrinsics.shape[0]
                n_ctx = self.view_sampler.num_context_views
                chunks = [
                    list(range(i, min(i + n_ctx, num_views)))
                    for i in range(0, num_views, n_ctx)
                ]
                # Pad the last chunk if it's smaller than n_ctx.
                if len(chunks[-1]) < n_ctx:
                    last = chunks[-1]
                    last += [last[-1]] * (n_ctx - len(last))
                iter_chunks = [(torch.tensor(c, dtype=torch.int64), torch.tensor(c, dtype=torch.int64)) for c in chunks]
            else:
                iter_chunks = None

            for_range = iter_chunks if self.stage == "test" else range(times_per_scene)
            for item in for_range:
                if self.stage == "test":
                    context_indices, target_indices = item
                else:
                    try:
                        context_indices, target_indices = self.view_sampler.sample(
                            scene,
                            extrinsics,
                            intrinsics,
                        )
                    except ValueError:
                        continue

                context_images = images[context_indices]
                target_images = images[target_indices]

                expected_shape = (3, *self.cfg.image_shape)
                context_bad_shape = tuple(context_images.shape[1:]) != expected_shape
                target_bad_shape = tuple(target_images.shape[1:]) != expected_shape
                if self.cfg.skip_bad_shape and (context_bad_shape or target_bad_shape):
                    continue

                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self._get_bound(self.cfg.near, len(context_indices)),
                        "far": self._get_bound(self.cfg.far, len(context_indices)),
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self._get_bound(self.cfg.near, len(target_indices)),
                        "far": self._get_bound(self.cfg.far, len(target_indices)),
                        "index": target_indices,
                    },
                    "scene": scene,
                }

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)

                if self.cfg.image_shape == list(context_images.shape[2:]):
                    yield example
                else:
                    yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def __len__(self) -> int:
        if self.stage == "test":
            # Count non-overlapping context chunks across all scenes.
            import math
            total = 0
            n_ctx = self.view_sampler.num_context_views
            for scene_dir in self.scene_dirs:
                with (scene_dir / self.cfg.metadata_file).open("r") as f:
                    num_views = len(json.load(f)["frames"])
                total += math.ceil(num_views / n_ctx)
            return total
        return len(self.scene_dirs) * self.cfg.train_times_per_scene
