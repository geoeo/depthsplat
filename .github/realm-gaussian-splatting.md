# Realm datasets for Gaussian splatting: status and remaining work

Written 2026-10-16 against commit `c71e24a` on `main`. Audience: a coding agent continuing
this work in this repository. Items marked **verified** were checked in this workspace on
that date. Items marked **unverified** are inferred from reading the code and must be
tested before being relied on.

This file lives in `.github/` rather than `.github/agents/` because `.github/agents/` is
gitignored (`.gitignore:176`).

---

## 0. Summary

**Done.** Depth-only inference of the pretrained DepthSplat depth model on the realm
datasets is finished and robust: conversion scripts, a triplet pipeline with relative
poses, an in-process inference API, reference-depth bookkeeping, AOTInductor export, and
visualisation tooling all exist (section 3).

**Not started.** The Gaussian-splatting (GS) path has never been run on realm data in this
workspace: there is no GS output directory, no GS command in `.shell_history`, and no
evaluation index with held-out target views. All the code pieces exist upstream (GS encoder
head, CUDA rasterizer installed, PSNR/SSIM/LPIPS, PLY export) and two GS checkpoints are
present in `pretrained/`.

**What blocks a meaningful GS evaluation** (details in section 5):

1. **Metric settings not chosen.** The pipeline works in metres: every pose is expressed
   relative to the first camera, which becomes the identity (the old `--scale` flag was a
   debug option and is retired). `near`/`far` are fixed scene-wide in
   `config/dataset/custom_images.yaml` (currently 10 / 500 m); runs take them from there
   through `+experiment=realm` (`config/experiment/realm.yaml`). What is still open is which `gaussian_scale_max` value, in metres, suits
   ~80 m aerial scenes, and whether the models, trained on RealEstate10K/DL3DV at room
   scale, produce metrically accurate depth there. `TODO.md` ("Check depth/translation
   scaling") refers to this check.
2. **No held-out evaluation index.** The loader can take context/target splits from a JSON
   index, but the only existing index (`assets/evaluation_index_realm.json`) uses frames
   `[0,1,2]` as both context and target, so PSNR on it measures reprojection of inputs, not
   novel-view synthesis.
3. **No metric geometry reference except the OpenREALM DEM** in
   `custom/realm_2_100m_w_tandem/gnd_1/` (13 frames). The per-frame `dense/*.npy` maps are
   not metric and not multi-view consistent.
4. **No evaluation tooling** for rendered depth, DEM comparison, or per-scene score tables.

**Suggested order:** W1 smoke test → W2 scene builder with held-out targets → W3 metric
settings and accuracy check → W4 evaluation tooling → W5/W6 georeferencing and larger
reconstructions.

---

## 1. Environment (verified)

| Item | Value |
|---|---|
| Container | `.devcontainer/` compose service `ws_depthsplat`, repo at `/workspaces`, user `ubuntu` |
| Python | `/opt/conda/envs/depthsplat/bin/python` (auto-activated in login bash) |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| torch / CUDA | 2.14.1+cu132 on Ubuntu 24.04 / CUDA 13.2 (Dockerfile pins cu132 wheels), xformers 0.0.35 (cu130 build). Changed 2026-10-19 from 2.6.0+cu126; not yet re-verified on this machine, see `.github/agents/migration.md` |
| Rasterizer | `diff_gaussian_rasterization` imports fine (installed by `post-create-command.sh`) |
| Metrics deps | `lpips` OK, `scikit-image` OK, `tifffile` OK; `rasterio` and `open3d` **absent** |
| torch hub cache | `~/.cache/torch` does not exist. First model construction downloads DINOv2 via `torch.hub.load` (`src/model/encoder/unimatch/mv_unimatch.py:89`). GitHub is reachable. |
| Peak GPU memory, depth-only, 3 views at 480x640 | 4.78 GB (`outputs/*/metrics/peak_memory.json`) |
| `custom/` | Bind mount from the host (`.devcontainer/compose.yaml`). The host path was changed to `/home/marc/Datasets/AIT/situation_map_data_mount` and that change is **uncommitted**. Do not commit or revert it without asking. |
| Precision | `python -m src.main` runs TF32 (`src/main.py` `__main__`); `src/depth_inference.py` defaults to full fp32. The depth model is TF32-sensitive (up to ~1.3 m difference). Record which one each run used. |

Flags read at **import time** (must be in the process environment before Python starts):
`XFORMERS_DISABLED`, `TYPECHECK_DISABLED`. See `scripts/realm_triplet_depth.py` header and
`src/typecheck.py`.

Local-only (gitignored) agent docs that describe the depth pipeline in depth:
`.github/agents/realm-triplet-depth.agent.md`, `.github/agents/xformers-export-readiness.agent.md`.

---

## 2. The realm datasets in `custom/` (verified)

### 2.1 Common format

```
custom/<name>/
  imgs/<nanosecond-timestamp>.png     640x480 RGB
  intrinsics.txt                       "# fx fy cx cy width height" then one line
  kf_traj.txt                          <ts> r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

- Intrinsics are identical for every dataset: fx 415.991, fy 415.557, cx 322.739, cy 235.54.
  Horizontal FOV 75.1°, vertical 60.0°. Well under `max_fov: 120`.
- `kf_traj.txt` rows are 3x4 **camera-to-world** matrices (translation = camera position),
  OpenCV axes (+Z into the image), nadir-looking (third row ≈ `[0 0 -1]`). Translations are
  **UTM zone 33N metres** (E ≈ 5.9e5, N ≈ 5.28e6, Z = ellipsoidal/orthometric height ≈
  380–930 m). Never feed absolute UTM to the model: `load_trajectory()` keeps float64 and the
  loader casts to float32 only after poses are made relative to an anchor. To stay in metric
  units with small numbers, use the first camera pose as the reference and express every
  pose relative to it, so that the first camera becomes the identity matrix (see the note in
  W3).
- Keyframes are 0.6 s apart, median spacing 11–13 m. At ~77 m above ground the footprint is
  about 123 m x 92 m and the ground sampling distance is ~0.19 m/px, so consecutive keyframes
  overlap by roughly 90 %.

### 2.2 Catalogue

| Dataset | Images | Poses | Extras | Notes |
|---|---|---|---|---|
| `realm_1` | 163 | 163 | none | Different site from the others (UTM 589.9k/5275.2k, Z ≈ 887 m). Trajectory has >85 m jumps between consecutive keyframes at indices 51–52 and 129–138 (physically impossible at 0.6 s): treat those windows as pose outliers. |
| `realm_1_s` | 5 | 163 | none | Five frames of `realm_1` with the full trajectory file. Quick-test subset. |
| `realm_2` | 349 | 349 | none | Same flight as `realm_2_with_depth` (330 shared frame ids). Jumps >100 m at indices 95, 99, 101–104, 118, 120, 122–123. |
| `realm_2_with_depth` | 501 | 503 | `dense/` 501 x (npy, png, txt) | **Superset of `realm_2`, clean trajectory (no jumps). Recommended primary dataset.** Renamed from `realm_1_with_depth` on 2026-10-16 because it is the `realm_2` flight. |
| `realm_2_100m_w_tandem` | 13 | 13 | `gnd_1/` full OpenREALM stage output | Same flight as `realm_2`, different keyframe ids (no overlap with `realm_2_with_depth`). Only dataset with a metric DEM. Camera Z 480–484 m, DEM median 404.9 m → ~77 m above ground. Renamed from `realm_1_100m_w_tandem` on 2026-10-16. |

Also in `custom/`: `depthsplat.png`, `moge3.png`, `100m_depthsplat.png`, `100m_moge3.png`
are hand-made comparison figures. For frame `1781763302194161000` (tandem flight) MoGe's
metric depth reads ≈45–50 m while DepthSplat (metres, first camera as identity) reads ≈50–145 m (ground ≈85–100 m).
With ~77 m above ground, neither is obviously right; see W3. MoGe is not in this repo.

### 2.3 Reference geometry products

**`realm_2_with_depth/dense/dense_<ts>.npy`** — float32 480x640, ~0.4 % NaN holes. The
sibling `.txt` says `Scaling (Not Georeferenced)` with per-frame min/max; the `.png` is
16-bit linear (`npy = a*png + b`, a ≈ 6.9e-4, b = min). Median value ≈ 12.6 for a frame at
~80 m above ground (ratio ≈ 0.03). `scripts/evaluate_depth_aoti.py` docstring records that
these maps reproject between neighbouring views with r ≈ 0.03 versus r ≈ 0.87 for the model's
own output. **Not usable as metric ground truth**; at best a relative, per-frame
scale-aligned sanity check. The loader exposes them as `depth_reference` (copied next to
predictions as `<idx>_reference.npy`, older runs used `_gt.npy`).

**`realm_2_100m_w_tandem/gnd_1/`** — OpenREALM stage outputs for the 13-frame flight:

| Path | Content |
|---|---|
| `pose_estimation/air_6/trajectory/{kf_traj,gnss_traj,timestamps}.txt` | Identical to the top-level `kf_traj.txt`; GNSS trajectory has identity rotations. |
| `densification/dense/` | 8 frames of TANDEM dense depth, same `Not Georeferenced` scaling, range 6.8–326, `stage.log` mentions `Scale factor: 200`. Treat like the maps above. |
| `mosaicing/elevation/gtiff/elevation.tif` | **DEM**, float32, 2507 rows x 1648 cols, 0.15 m/px, EPSG:32633. GeoTIFF tags: tiepoint (33922) = pixel (0,0) at E 592007.1 / N 5285877.0; pixel scale (33550) = (0.15, 0.15); nodata = NaN. 16 % of pixels valid. Values 200–477 m, median 404.9; the ≈200 m tail looks like outliers and should be filtered. |
| `mosaicing/ortho/ortho.tif` + `ortho.yaml` | RGB orthophoto on the same grid; `roi: [592007.1, 5285501.1, 247.05, 375.9]` (x, y, w, h in metres) agrees with the GeoTIFF tags. |
| `ortho_rectification/ortho/ortho_00000{1..7}.png` | Per-frame rectified tiles. |

Geotransform without rasterio: `E = 592007.1 + 0.15 * col`, `N = 5285877.0 - 0.15 * row`.
`tifffile` (installed) or PIL (`Image.open(...).tag_v2`) both read the tags.

---

## 3. What already exists in the code (verified)

### 3.1 Scene conversion and depth inference

| File | Role |
|---|---|
| `scripts/convert_realm_to_custom_images.py` | Whole-scene converter for `realm_1` and `realm_2` only (hardcoded list, absolute `/workspaces` paths). Poses made relative to the first selected frame; has a `--scale` flag (divides translations) that was a debug option and is **retired: leave it at the default 1**. Writes `datasets/custom_images/test/<name>/cameras.json` and an index JSON with context = every N-th frame and **target = all frames (includes the context views)**. Exposes `load_intrinsics()` and `load_trajectory()`. |
| `scripts/realm_triplet_depth.py` | Triplet pipeline: `preprocess()` builds one scene per triplet under `datasets/custom_images_triplets/test/<dataset>_off<o>_t<k>_<i>-<j>-<l>/`, poses relative to the triplet's first frame (float64, identity check with 1e-4 tolerance because UTM translations are ~1e6), records `scale` (debug flag, retired, always 1), `offset`, `indices`, `frame_id`, `source_index`, `depth_reference`, then runs inference in-process. **Does not record the anchor pose**, so outputs cannot be mapped back to UTM without re-reading `kf_traj.txt`. `--output-dir` must be absolute (`relative_to(REPO_ROOT)`). |
| `src/depth_inference.py` | `compose_config()`, `depth_inference_overrides()` (uses `dataset/view_sampler=all`, so every frame is context and no index is needed), `run_depth_inference(cfg, full_fp32=True)`. |
| `scripts/depth_export_common.py`, `export_dataset.py`, `aot_compile_depth.py`, `run_depth_aoti.py`, `evaluate_depth_aoti.py` | AOTInductor export path for the depth predictor (`DepthExport.forward` returns `depth_preds[-1]`). Not needed for GS, but `prepare_scenes()` shows how to reuse the triplet builder. `--dataset` choices there omit `realm_2_100m_w_tandem`. |
| `scripts/visualize_depth_npy.py` | Matplotlib viewer for `<out>/images/<scene>/depth/*.npy` plus `_cam.png`/`_reference.npy`. |
| `scripts/compare_depth_to_reference.py` | Least-squares **scale-aligned** depth comparison. Fine for cross-model comparison, wrong for A/B of identical weights. |
| `scripts/inference_depth_realm.sh`, `run_vits_experiment.sh`, `.vscode/launch.json` | Known-good Hydra override sets for depth-only runs on `custom_images`. The shell scripts use `+experiment=re10k dataset=custom_images ...` and re-pass `image_shape`/`near`/`far` read from the yaml; the two "Depth Inference" launch entries use `+experiment=realm` (section 3.3) and pass none of them. |

Existing outputs: `outputs/depthsplat-depth-base-realm_{1,2}` (3 frames each),
`outputs/depthsplat-depth-base-realm_1_with_depth-triplets` (30 triplets from offset 70; produced before the rename, so the directory keeps the old name),
`outputs/so_depths`, `outputs/aoti/`. `datasets/custom_images/test/realm_{1,2}` currently hold
3-frame scenes at scale 1.0, offset 0.

### 3.2 Dataset loader: `src/dataset/dataset_custom_images.py`

- Scene = directory containing `cameras.json` (`metadata_file`) with `frames[]` of
  `{image, intrinsics (3x3 pixels), extrinsics (4x4 C2W), frame_id?, depth_reference?}`.
  `intrinsics_are_normalized: false` → divided by W/H in the loader. Images are resized to
  `image_shape` if they differ.
- `scene_selection` filters by directory name; the yaml default is `realm_1`, so pass
  `dataset.scene_selection=null` to run every scene.
- **Test-stage behaviour (`__iter__`)** depends on the view sampler:
  - `view_sampler.num_context_views > 0` → frames are chunked into non-overlapping windows of
    that size and **target == context** (depth-only use case, no novel views).
  - `== 0` → `view_sampler.sample()` decides: `all` returns all frames as context and target;
    `realm`/`evaluation` (`ViewSamplerEvaluation`) looks the scene name up in `index_path`
    JSON and returns the listed `context`/`target` index lists. This is the only path that
    gives held-out targets. `assets/evaluation_index_realm.json` is keyed by the `scene`
    string in `cameras.json`.
- `near`/`far` come from the dataset config and are broadcast to every view. They are a
  property of the dataset config, not of a scene or a frame: one pair, taken from
  `config/dataset/custom_images.yaml`, is applied to the whole scene (all context and
  target views). `cameras.json` does not carry them. The encoder uses them as the
  inverse-depth candidate range (`min_depth = 1/far`, `max_depth = 1/near`), and the
  rasterizer uses them as clip planes.

### 3.3 Configs

- `config/dataset/custom_images.yaml`: 480x640, `near 10`, `far 500`, `view_sampler: realm`.
  The yaml is the single source of truth for `image_shape`, `near` and `far`; they apply to
  the whole scene. If they ever need to change, change the yaml so depth-only and GS runs
  stay comparable.
- `config/experiment/realm.yaml` (added 2026-10-16): the way to use the yaml values with
  `src.main`. It carries the same `model`/`loss` block as `experiment/re10k.yaml` (the
  checkpoints require it; verified identical after composition) plus `trainer.num_nodes`,
  and sets **no** dataset values, so `custom_images.yaml` wins. With `+experiment=re10k`
  instead, the experiment's own `image_shape: [256,256]`, `near: 0.5`, `far: 100` are merged
  last and win (verified by composing both), which is why the older shell scripts re-pass
  the yaml values explicitly. **Deleting `dataset.near`/`dataset.far` from a command that
  uses `+experiment=re10k` does not restore the yaml values; switch to `+experiment=realm`.**
  Composition and the typed-config load were verified for both launch entries and for the
  GS override set in section 6; a GPU run was not.
- `config/dataset/view_sampler/realm.yaml`: `ViewSamplerEvaluation` with
  `index_path: assets/evaluation_index_realm.json`, `num_context_views` defaults to 0.
- Extra keys set by an experiment that the custom dataset dataclass lacks (`make_baseline_1`,
  `highres`, `min_views`, `test.dec_chunk_size`, ...) are tolerated: `src/config.py:73-76`
  uses dacite without strict mode. `+experiment=realm`, and `+experiment=re10k` +
  `dataset=custom_images`, are verified to compose and type-check; `+experiment=dl3dv` +
  `dataset=custom_images` is unverified.
- **Patch shim.** `EncoderDepthSplat.get_data_shim()`
  (`src/model/encoder/encoder_depthsplat.py:379-385`) centre-crops every batch to a multiple
  of `shim_patch_size * downscale_factor`. With `+experiment=re10k` or `+experiment=realm` that is 16 px (`shim_patch_size` 4, verified) → 480x640 is
  untouched. With `+experiment=dl3dv` (`shim_patch_size: 16`) it is 64 px → images become
  **448x640** at test time, intrinsics adjusted, and saved depth/colour are 448x640.

### 3.4 Gaussian-splatting path in `src/model/model_wrapper.py::test_step` (line 362)

1. Encoder forward (`src/model/encoder/encoder_depthsplat.py::forward`). With
   `train_depth_only=false` it builds Gaussians: one per pixel per context view, means =
   unprojected depth, colour SH initialised from the input image, opacities from the
   regressor. For `v > 3` context views the cost volume only matches each view against its
   `local_mv_match = 2` nearest cameras (by translation distance).
   **Gaussian scales are clamped in world units**:
   `clamp(softplus(raw - 4), gaussian_scale_min, gaussian_scale_max)`
   (`src/model/encoder/common/gaussian_adapter.py`). They are not depth-relative, so for metric
   aerial scenes `gaussian_scale_max` must be chosen in metres (W3). README's
   `gaussian_scale_max=0.1` recommendation for the 448x768 models assumes the room-scale
   training scenes (default is 3.0).
2. `test.save_gaussian=true` → `save_gaussian_ply()` writes `<output_dir>/gaussians/<scene>.ply`
   with an 8-px border trimmed, **expressed in the first context camera's frame** (`export_ply`
   rotates means by the inverse of that camera's rotation). Viewable in SuperSplat.
3. Decoder (`DecoderSplattingCUDA`) renders the **target** views with `depth_mode=None`, so no
   rendered depth is produced. `render_depth()` exists with modes
   `depth | disparity | relative_disparity | log` (`src/model/decoder/cuda_splatting.py:222`)
   but is not wired into `test_step`.
4. `test.save_depth=true` in the GS path saves the **encoder depth of the context views**
   (`visualization_dump["depth"]`) to `images/<scene>/depth/<idx>.{png,npy}` plus `_cam.png`
   and `_reference.npy`. At test time `depth_preds` has exactly one entry (the intermediate
   bilinear depth is only appended when `self.training`, `mv_unimatch.py:550-557`), so index
   0 is the final prediction. Verified by reading the code.
5. `test.save_image=true` → `images/<scene>/color/<target_idx>.png`, with `save_gt_image=true`
   also `<target_idx>_gt.png`. `test.save_video=true` → `videos/<scene>_frame_<ctx>.mp4`.
6. `test.compute_scores=true` → PSNR/SSIM/LPIPS(vgg) per batch, averaged in `on_test_end()`,
   written to `<output_dir>/metrics/scores_{psnr,ssim,lpips}_all.json` (one number per batch,
   in dataloader order, **no scene names**) and `scores_all_avg.json`, plus `benchmark.json`
   and `peak_memory.json`.
7. Checkpoint loading in test mode (`src/main.py:235-243`): `checkpointing.pretrained_model`
   is loaded into the whole `ModelWrapper` with `strict=True` unless
   `checkpointing.no_strict_load=true`. The LPIPS loss stores its weights as non-persistent
   buffers (`src/loss/loss_lpips.py`), so they are absent from the state dict and strict
   loading is expected to work.

### 3.5 Checkpoints in `pretrained/` (verified by loading on CPU)

| File | Kind | Backbone | Views | Notes |
|---|---|---|---|---|
| `depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth` | GS, full model | ViT-B (768-d, 12 blocks) | 2–6 | `state_dict` with keys only under `encoder.*` (`depth_predictor`, `feature_upsampler`, `gaussian_regressor`, `gaussian_head`). `gaussian_head` outputs 37 channels = 1 opacity + 2 offset + 3 scale + 4 rotation + 27 SH (degree 2). Requires `num_scales=2 upsample_factor=4 lowest_feature_resolution=8 monodepth_vit_type=vitb` (README). **Primary candidate.** |
| `depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth` | GS, full model | ViT-S (384-d) | 4–10 | README uses `upsample_factor=8 lowest_feature_resolution=8` (num_scales 1, vits). Candidate for longer windows. |
| `depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth` | GS, full model | ViT-S | 2 | 256x256 indoor model, low priority. |
| `depthsplat-depth-base-352x640-randview2-8-65a892c5.pth` | Depth only | ViT-B | 2–8 | `{"model": ...}` format, loaded into `encoder.depth_predictor`. The model used in all realm runs so far. |
| `depthsplat-depth-small-352x640-randview2-8-e807bd82.pth` | Depth only | ViT-S | 2–8 | Used by `run_vits_experiment.sh`. |

Upstream RE10K test chunks exist in `datasets/re10k/test/` (two scenes), useful to confirm the
GS path works at all before blaming realm data.

---

## 4. Known pitfalls (verified unless stated)

- `scripts/realm_triplet_depth.py --output-dir` must be absolute. `.vscode/launch.json`
  passes `--output`, which argparse accepts only as an abbreviation of `--output-dir`.
- Any index JSON whose `target` list contains the context indices inflates PSNR/SSIM/LPIPS
  (the model re-renders its inputs). `build_eval_index()` in the converter does exactly that.
- `realm_1` and `realm_2` trajectories contain outlier jumps (section 2.2). Window selection
  must skip them, or use `realm_2_with_depth`.
- `dataset.scene_selection` defaults to `realm_1` in the yaml. Pass `null` or a scene name.
- `near`/`far` typed into a command line silently diverge from the yaml (and from every
  depth-only run that read the yaml), and simply deleting them from a `+experiment=re10k`
  command makes Hydra fall back to the experiment's 0.5 / 100, not to the yaml (verified).
  Use `+experiment=realm`, which takes the one pair from `config/dataset/custom_images.yaml`
  and applies it to the whole scene. The two "Depth Inference" entries in
  `.vscode/launch.json` were switched to it on 2026-10-16 (they previously hardcoded
  `dataset.far=1000.0`, which disagreed with the yaml's 500).
- Hydra `+experiment=dl3dv` crops realm images to 448x640 (section 3.3). Decide explicitly.
- `test.save_depth` in the GS path saves encoder depth at context views, not rendered depth at
  target views.
- Scores are averaged over batches without scene ids; recover the order from the sorted scene
  directory names (the loader iterates `sorted(search_root.iterdir())`, filtered by
  `scene_selection`, one batch per scene when using the `realm` sampler and
  `data_loader.test.num_workers=4` — the worker split `i % num_workers == id` can reorder
  output across workers, so set `data_loader.test.num_workers=0` when order matters).
  (**unverified**: the reordering claim; the modulo split is in `__iter__`.)
- TF32 vs fp32 moves depth by up to metres; `src/main.py` has no flag to switch, only
  `src/depth_inference.py` does.
- 12 GB GPU: depth-only at 3 views used 4.8 GB. GS with 4–6 context views at 480x640 plus
  LPIPS-VGG is **unverified**; `test.render_chunk_size` only chunks target rendering.
- `convert_to_buffer` makes LPIPS weights non-persistent, but `lpips` still downloads VGG
  weights from torchvision on first use (network).
- PLY export is in the first-context-camera frame, not UTM, and discards an 8-px border.
- Output directories are rebuilt from scratch by `preprocess()` (`rmtree` of
  `<data_root>/test`), so use a separate `--data-root` per experiment type.

---

## 5. Remaining work

Priorities: P0 = needed for any GS result, P1 = needed for a credible evaluation,
P2 = valuable extensions, P3 = housekeeping. Sizes: S < 1 h, M = hours, L = a day or more.

### W1 (P0, S) — Smoke-test the GS model on a realm scene

Goal: prove that the full model loads strictly, renders, writes a PLY, and produces scores
on realm data. Use the existing converter (section 6 has the exact commands). Acceptance:

- `Loaded pretrained weights: pretrained/depthsplat-gs-base-...` printed, no missing/unexpected keys.
- `outputs/<run>/gaussians/<scene>.ply`, `images/<scene>/color/*.png` and `*_gt.png`,
  `images/<scene>/depth/*.npy`, `metrics/scores_all_avg.json` exist.
- Renders at the context views look like the inputs (PSNR well above 20 dB on those; these
  are not held-out).
- Record peak memory (`metrics/peak_memory.json`) with 4 context views at 480x640.

If composition with `+experiment=re10k` fails on an unexpected key, compare against the
launch configurations in `.vscode/launch.json`, which are known to compose for the depth path.

### W2 (P0, M) — Scene builder for GS windows with held-out targets

Generalise `realm_triplet_depth.py::build_triplet_scene` into a new script
(suggested `scripts/realm_gs_scenes.py`) that:

- Takes `--dataset` (any folder under `custom/`, including `realm_2_100m_w_tandem`),
  `--offset`, `--window N`, `--stride`, `--num-windows`, `--context-step k` (context = every
  k-th frame of the window, targets = the rest), `--data-root`
  (default a new root, e.g. `datasets/custom_images_gs`), `--index-path`.
  **No `--scale` option**: units are metres throughout.
- Makes poses relative to the window's first context frame in float64 (reuse
  `se3_inverse`, keep the identity assertion). No rescaling.
- **Adds to `cameras.json`:** `anchor_c2w_utm` (4x4, float64 as written by json),
  `units: "m"`, `utm_epsg: 32633`, `context_indices`, `target_indices`, and per-frame
  `frame_id`, `source_index`, `depth_reference`. Without the anchor nothing can be
  georeferenced later.
- Does **not** write `near`/`far` into `cameras.json` or the index, and takes no
  `--near`/`--far` flags: the pair in `config/dataset/custom_images.yaml` applies to the
  whole scene, and the inference side reads it from there (as `realm_triplet_depth.py`
  does via `DATASET_CFG_PATH`).
- Writes the evaluation index with **disjoint** `context`/`target` lists keyed by the scene
  name, and optionally a second index entry per scene with `target == context` for
  input-view reprojection checks.
- Refuses or flags windows that contain a trajectory jump (step > 40 m between consecutive
  keyframes) or a frame without a pose.
- Prints a table: scene, source indices, baseline in metres between first and last context
  camera, index file.

Suggested first protocol on `realm_2_with_depth`: windows of 7 consecutive keyframes,
context `[0,2,4,6]` (~40 m baseline end to end), targets `[1,3,5]` (interpolation); also an
extrapolation variant with context `[0,1,2,3]`, targets `[4,5,6]`.

### W3 (P0, M) — Metric-unit convention, depth-range settings, and accuracy check

**Unit convention (fixed, not a tuning knob).** The pipeline works in metres. For smaller
metric units, take the first camera pose of the scene (or window) as the reference and
transform all poses relative to it, `T_rel = inv(T0) @ T` with the SE(3) inverse
`[R^T | -R^T t]`. After this transformation the first camera pose must be the identity
matrix (assert it, with a loose tolerance of ~1e-4 because the UTM translations are ~1e6
and the subtraction leaves mm-level residuals). Every translation and every predicted depth
then stays in metres while the large UTM offsets that float32 cannot represent are removed.
`build_triplet_scene()` in `scripts/realm_triplet_depth.py` and `convert_dataset()` in
`scripts/convert_realm_to_custom_images.py` already do exactly this, and W2 must keep it.
Keep `T0` (the reference pose, in UTM) in `cameras.json` so results can be placed back into
UTM. The `--scale` flag in those two scripts (divide translations by a factor) was a debug
option and is retired: leave it at 1, do not add it to new scripts, and do not plan
experiments around it.

**`near` / `far` (fixed by the yaml, applied to the whole scene).** The pair in
`config/dataset/custom_images.yaml` (currently `near: 10`, `far: 500`, metres) is the only
source; it is applied to every view of every scene, and both the depth-only and the GS
runs take it from there (`+experiment=realm`, or the yaml read in
`scripts/realm_triplet_depth.py` and `scripts/inference_depth_realm.sh`) rather than from
the command line. The cost volume places
128 inverse-depth candidates between `1/far` and `1/near`, so the only thing to verify is
that the pair brackets the scenes: for the tandem flight the ground is ~77 m below the
camera at nadir and ~97 m at the image corners, trees are 10–20 m closer, and 10 / 500
brackets that with room to spare. If a different pair is ever justified, change the yaml
(one change for all runs), not individual commands.

**What remains to decide, in metres.** One setting:

- `gaussian_scale_max`: a splat covering about one pixel at 80 m is ~0.2 m (GSD 0.19 m/px).
  README's 0.1 was meant for room-scale scenes; the config default is 3.0. Candidates: 0.5,
  1.0, 3.0.

**Accuracy check (do this first, needs no rendering).** The models were trained on
room-scale scenes, so metric accuracy at 80 m is not guaranteed. For each tandem keyframe,
look up the DEM height at the camera's UTM (E, N), compute `AGL = cam_z - DEM`, and compare
with the predicted depth at the principal point. Thirteen numbers settle whether DepthSplat
(~85–100 m in the existing run) or MoGe (~45–50 m) is closer to metric for that flight, and
whether the depth model is usable in metres at all.

**Plan.** Fix one window set from W2, run the three `gaussian_scale_max` candidates above
with `near`/`far` from the yaml, and compare (a) held-out PSNR/SSIM/LPIPS, (b) predicted
nadir depth against the DEM-derived height above ground (W4c-lite above), (c) the PLY
visually. Also run the depth-only model on the same windows so depth quality and GS quality
can be separated. Write the chosen `gaussian_scale_max` into the GS launch script or
`.vscode/launch.json` entry and into this file.

### W4 (P1) — Evaluation tooling

- **W4a (S)** Score table script: join `scores_*_all.json` with scene names and target
  indices, emit a per-scene Markdown/CSV table with mean ± std, plus the run manifest
  (near/far, `gaussian_scale_max`, context count, image shape, precision, checkpoint, git
  commit). Write the
  manifest at run time into `<output_dir>/manifest.json`.
- **W4b (M)** Rendered depth at target views: add a `test.save_rendered_depth` flag in
  `test_step` that calls `self.decoder.forward(..., depth_mode="depth")` and saves
  `images/<scene>/depth_rendered/<target_idx>.npy`; or add `src/gs_inference.py` mirroring
  `src/depth_inference.py` (in-process, `full_fp32` switch, no Hydra CLI) and do it there.
  The in-process route is preferable because it also fixes the precision choice.
- **W4c (M)** DEM comparison for `realm_2_100m_w_tandem`: unproject predicted or rendered
  depth with K and the absolute pose (`anchor_c2w_utm @ relative_c2w`; everything is already
  in metres) into UTM, sample `elevation.tif` with the geotransform in section 2.3 (nearest or bilinear,
  skip NaN and values < 300 m), and report height error statistics (median absolute, RMSE,
  fraction within 1 m / 3 m), per frame and overall. Optionally rasterise Gaussian means or
  unprojected points into a 0.15 m height map and difference it against the DEM. Watch
  float32: subtract a local origin (e.g. the anchor translation) before any float32 math and
  keep the origin in the manifest.
- **W4d (S)** MoGe comparison: if MoGe outputs are provided as `.npy`, reuse
  `compare_depth_to_reference.py` for the scale-aligned view and add a `--no-align` flag for
  the metric view.
- **W4e (M)** Multi-view depth consistency (self-supervised): warp depth from view i to view j
  with the known poses and report correlation / inlier ratio. The r ≈ 0.87 vs 0.03 figures in
  `evaluate_depth_aoti.py` came from such a script, which is not in the repo.

### W5 (P2, M) — Georeferenced Gaussian export

Write a PLY (and optionally a point cloud of means) in UTM: undo `export_ply`'s rotation into
the first camera, apply `anchor_c2w_utm` (positions and scales are already in metres),
subtract a local origin stored in a sidecar JSON (UTM magnitudes break float32 PLY), and keep the full
frame (no 8-px trim) behind a flag. This makes the splats comparable with the orthophoto and
DEM in GIS tools.

### W6 (P2, L) — Longer trajectories

Reconstruct a whole flight segment: either the small 4–10-view model, or sliding windows with
all windows anchored to one global frame (not per-window anchors) so their Gaussians can be
concatenated. Needs W5 and a merge/deduplication strategy; memory on the 12 GB GPU is the
constraint.

### W7 (P3, S) — Housekeeping

- `.vscode/launch.json`: `--output` → `--output-dir` in "Realm Triplet Depth"; add a GS launch
  config once W1 works.
- Optionally switch `scripts/inference_depth_realm.sh`, `scripts/run_vits_experiment.sh` and
  `depth_inference_overrides()` in `src/depth_inference.py` to `+experiment=realm`, so they
  stop re-reading the yaml and re-passing `image_shape`/`near`/`far` themselves. Not
  required: they already produce the yaml values.
- `scripts/depth_export_common.py::add_common_args`: widen `--dataset` choices or drop the
  restriction.
- `TODO.md`: "Check why realm inference wont work with vitl" is still open and unrelated to
  GS (no ViT-L checkpoint is present anyway).
- Done 2026-10-16: the folders were renamed on the host mount (`realm_1_with_depth` →
  `realm_2_with_depth`, `realm_1_100m_w_tandem` → `realm_2_100m_w_tandem`) and every
  script, launch entry and doc now uses the new names. Generated data produced before the
  rename keeps the old names and should be regenerated rather than renamed:
  `outputs/depthsplat-depth-base-realm_1_with_depth-triplets`,
  `datasets/custom_images_triplets/test/realm_1_with_depth_off0080_*`, and the scene names
  inside `outputs/dataset_cfg_with_gt_m.pt2`.

---

## 6. Smoke-test recipe for W1 (commands unverified end to end)

All paths relative to `/workspaces`; run from there in the `depthsplat` env.

Step 1 — build a 7-frame scene for `realm_1` and `realm_2` with context every 2nd frame
(targets will include the context frames; acceptable for a smoke test only):

```bash
python scripts/convert_realm_to_custom_images.py --offset 70 --num-frames 7 --context-step-size 2
```

This rewrites `datasets/custom_images/test/realm_{1,2}/` and
`assets/evaluation_index_realm.json` (context `[0,2,4,6]`, target `[0..6]`). Poses are in
metres relative to frame 70 (identity); do not pass `--scale`.

Step 2 — run the base GS model. Model overrides follow the README's 448x768 base-model
command; `+experiment=realm` takes `image_shape`, `near` and `far` from
`config/dataset/custom_images.yaml` (section 3.3). This override set composes and
type-checks (verified); the run itself has not been executed:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=realm \
  dataset.roots=[datasets/custom_images] \
  dataset.scene_selection=realm_2 dataset.test_chunk_interval=1 \
  dataset/view_sampler=realm dataset.view_sampler.index_path=assets/evaluation_index_realm.json \
  mode=test \
  model.encoder.num_scales=2 model.encoder.upsample_factor=4 \
  model.encoder.lowest_feature_resolution=8 model.encoder.monodepth_vit_type=vitb \
  model.encoder.gaussian_adapter.gaussian_scale_max=1.0 \
  checkpointing.pretrained_model=pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth \
  test.compute_scores=true test.save_image=true test.save_gt_image=true \
  test.save_gaussian=true test.save_depth=true test.save_depth_npy=true \
  data_loader.test.num_workers=0 wandb.mode=disabled \
  output_dir=outputs/depthsplat-gs-base-realm_2-smoke
```

Expected outputs under `outputs/depthsplat-gs-base-realm_2-smoke/`: `gaussians/realm_2.ply`,
`images/realm_2/color/00000{0..6}.png` and `_gt.png`, `images/realm_2/depth/`, `metrics/`.

Things most likely to fail, in order: (1) strict state-dict mismatch if the encoder
architecture overrides differ from the checkpoint (composition itself is verified), (2) CUDA
OOM (drop to 3 context views), (3) first-run downloads of DINOv2 and LPIPS-VGG weights.

Depth-only reference for the same window, for comparison:

```bash
python scripts/realm_triplet_depth.py --dataset realm_2_with_depth --offset 70 --count 6 \
  --output-dir /workspaces/outputs/depthsplat-depth-base-realm_2_with_depth-off70
```

(Both runs take `near`/`far` from `config/dataset/custom_images.yaml`, the GS run through
`+experiment=realm` and the triplet script by reading the yaml, so they match by
construction and apply to the whole scene. `gaussian_scale_max` is given in metres and is
the W3 knob; 1.0 is a starting point.)

---

## 7. Proposed evaluation protocol (to implement via W2–W4)

| Axis | Measure | Data | Tooling |
|---|---|---|---|
| Novel-view synthesis | PSNR / SSIM / LPIPS on held-out targets, interpolation and extrapolation splits | `realm_2_with_depth` windows (clean trajectory), several offsets covering fields, forest edge, buildings | in-repo metrics + W4a |
| Geometry, metric | Height error of unprojected depth / Gaussian means vs DEM | `realm_2_100m_w_tandem` (13 frames) | W4c |
| Geometry, relative | Scale-aligned error vs MoGe (external) and vs OpenREALM dense (relative only) | any | `compare_depth_to_reference.py`, W4d |
| Geometry, self-consistency | Cross-view depth reprojection agreement | any | W4e |
| Runtime | encoder / decoder seconds, peak memory | all runs | `metrics/benchmark.json`, `peak_memory.json` |
| Qualitative | PLY in SuperSplat, colour renders, depth maps | all runs | `visualize_depth_npy.py`, W5 |

Always log: dataset, offset, window, context/target indices, near/far,
`gaussian_scale_max`, image shape (and whether the patch shim cropped), checkpoint,
precision (TF32 or fp32), xformers state, git commit.

---

## 8. Questions for the user

1. Which site matters: the `realm_1` site (Z ≈ 887 m) or the `realm_2`/`realm_2_with_depth`
   flight (Z ≈ 470 m)? Only the latter has a DEM.
2. Can MoGe depths be exported as `.npy` for the tandem frames (the figures in `custom/`
   suggest MoGe was run elsewhere)?
3. Can OpenREALM produce `gnd_1`-style output (DEM/orthophoto) for the whole
   `realm_2_with_depth` flight, or a georeferenced dense depth, so geometry can be scored on
   more than 13 frames?
4. Preference between the base 2–6-view model and the small 4–10-view model for longer
   windows?
5. Should the uncommitted `compose.yaml` host-path change be committed?

---

## Appendix: verified numbers

- Trajectory extents (UTM metres): `realm_1` 547 x 387 x 106; `realm_2` 533 x 375 x 97;
  `realm_2_with_depth` 619 x 416 x 115; `realm_2_100m_w_tandem` 50 x 157 x 4.
- Predicted depth in metres (triplets from `realm_2_with_depth` offset 70): per-view min/max
  roughly 38–183 m, means 48–89 m. Reference `dense` maps for the same frames: means 19–28
  (unitless).
- DEM: `elevation.tif` shape (2507, 1648), valid 16.4 %, origin E 592007.1 / N 5285877.0,
  0.15 m/px; nodata NaN.
- Depth-only run, 3 views, 480x640, ViT-B: peak 4.78 GB GPU memory.
