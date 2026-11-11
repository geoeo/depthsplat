# Realm datasets for Gaussian splatting: status, results and remaining work

Last updated 2026-10-16 (work on top of commit `c71e24a`, `main`, uncommitted). Audience: a
coding agent continuing this work. Items marked **verified** were run in this workspace on
that date; **unverified** items are inferred from code and must be tested before relying on
them.

This file lives in `.github/` rather than `.github/agents/` because `.github/agents/` is
gitignored.

**Reference policy.** The OpenREALM products shipped with the datasets, the per-frame
`dense/*.npy` depth maps and the `elevation.tif` DEM, come from an *earlier reconstruction*
of the same flights and can be wrong (holes, outliers, forest canopy, mosaic borders). They
are references, never ground truth. Every comparison against them in this document and in
the scripts is reported as a *disagreement* between two reconstructions. The metrics that do
not depend on the old reconstruction, held-out novel-view scores and cross-view depth
consistency, are the primary ones. (The camera poses in `kf_traj.txt` also come from
OpenREALM; there is no independent pose reference.)

---

## 0. Summary

**State.** The full Gaussian-splatting (GS) path runs on realm data end to end with metric
inputs and metric outputs, and every evaluation tool planned in the first version of this
document exists and has been exercised (sections 3 and 4). Only W6 (whole-trajectory
reconstruction) is untouched.

**Key finding (verified).** The pretrained GS checkpoints were trained on room-scale scenes
and collapse to the far plane when given metric aerial cameras, whatever `near`/`far` is
used. Presenting the same scene at room scale fixes it. The encoder therefore has an opt-in
scene-scale normalisation (metric in, metric out) that is on by default in the GS runner.
Its factor is derived from the scene-wide `near`/`far` of `config/dataset/custom_images.yaml`
(`sqrt(near·far)` → 3 units), because the alternative rule based on keyframe spacing
collapsed on a slow flight segment. See section 4.

**Second finding (verified).** A hybrid that loads the GS checkpoint and then overwrites its
depth predictor with the metric depth-only checkpoint needs no normalisation and gives more
self-consistent geometry on every hard window, at a small cost in image metrics (16.0 vs
16.9 dB). Which one to use depends on whether images or geometry matter downstream; this is
a user decision (section 7).

**Headline numbers (5 windows of 7 keyframes, context `[0,2,4,6]`, 15 held-out target
views, 480x640, full fp32):** normalised GS 16.87 ± 2.78 dB PSNR, SSIM 0.334, LPIPS 0.474;
hybrid 16.02 ± 2.66 dB, SSIM 0.325, LPIPS 0.526. On the one window with usable reference-DEM
coverage, nadir depth and DEM height above ground agree within 4–22 % on open fields.

**Remaining work** is listed in section 5; the largest items are whole-trajectory
reconstruction (W6), the choice between normalised GS and hybrid, and a better consistency
metric.

---

## 1. Environment (verified)

| Item | Value |
|---|---|
| Container | `.devcontainer/` compose service `ws_depthsplat`, repo at `/workspaces`, user `ubuntu` |
| Python | `/opt/conda/envs/depthsplat/bin/python` (auto-activated in login bash) |
| GPU | NVIDIA GeForce RTX 3060, 12 GB |
| torch / CUDA | 2.6.0+cu126, xformers 0.0.29.post3, `diff_gaussian_rasterization` installed, `lpips`, `scikit-image`, `tifffile`; `rasterio` and `open3d` absent (not needed) |
| torch hub cache | Now populated (`~/.cache/torch/hub`: DINOv2 ViT-B weights, VGG16 for LPIPS). A fresh container downloads them on first run; GitHub is reachable. |
| Peak GPU memory | depth-only, 3 views: 4.8 GB. GS, 4 context + 3 target views, fp32, LPIPS, PLY and rendered depth: 6.3 GB (`outputs/gs-eval-realm/metrics/peak_memory.json`). |
| `custom/` | Bind mount from the host (`.devcontainer/compose.yaml`; host path change is **uncommitted**, do not commit or revert without asking). |
| Precision | `python -m src.main` runs TF32; `src/depth_inference.py` and `src/gs_inference.py` default to full fp32. All numbers in section 4 are fp32 unless stated. |

Import-time flags: `XFORMERS_DISABLED`, `TYPECHECK_DISABLED` (set in the environment before
Python starts; `realm_gs_eval.py` and `realm_triplet_depth.py` pre-parse `--xformers`).

---

## 2. Datasets in `custom/` (verified)

Format: `imgs/<ns-timestamp>.png` (640x480 RGB), `intrinsics.txt` (fx 415.991, fy 415.557,
cx 322.739, cy 235.54, identical for all datasets; HFOV 75.1°, VFOV 60.0°), `kf_traj.txt`
(timestamp + 3x4 camera-to-world, OpenCV axes, nadir-looking, UTM zone 33N metres, EPSG
32633). Never feed absolute UTM to the model: `load_trajectory()` keeps float64 and poses are
made relative to an anchor camera (identity) before the loader casts to float32.

| Dataset | Images | Poses | Extras | Notes |
|---|---|---|---|---|
| `realm_1` | 163 | 163 | none | Different site (UTM 589.9k/5275.2k, Z ≈ 887 m). Trajectory jumps >85 m between consecutive keyframes at indices 51–52 and 129–138 (pose outliers). |
| `realm_1_s` | 5 | 163 | none | Five frames of `realm_1`. |
| `realm_2` | 349 | 349 | none | Same flight as `realm_2_with_depth` (330 shared frame ids). Jumps >100 m at indices 95, 99, 101–104, 118, 120, 122–123. |
| `realm_2_with_depth` | 501 | 503 | `dense/` 501 x (npy, png, txt) | Superset of `realm_2`, clean trajectory. **Primary dataset.** Renamed from `realm_1_with_depth` on 2026-10-16. |
| `realm_2_100m_w_tandem` | 13 | 13 | `gnd_1/` OpenREALM stage output incl. reference DEM | Same flight, different keyframe ids. Renamed from `realm_1_100m_w_tandem`. |

Flight profile of `realm_2_with_depth` (camera Z from `kf_traj.txt`): climb from 382 m
(frame 0) through 428–469 m (frames 70–97) to ~480–495 m (frames 200+; the tandem keyframes
are at 480–484 m). The reference DEM puts the open fields at ~395–410 m, so the windows used
below are 30–70 m above ground early in the flight and ~75–90 m later. Keyframes are 0.6 s
apart, median spacing 11–13 m; consecutive keyframes overlap by roughly 90 %.

Reference products (see the policy above):

- `realm_2_with_depth/dense/dense_<ts>.npy`: float32 480x640, "Scaling (Not Georeferenced)"
  per the sibling `.txt`, median ≈ 12.6 for a frame ~80 m above ground. Not metric, not
  multi-view consistent. The loader exposes them as `depth_reference`, copied next to
  predictions as `<idx>_reference.npy`.
- `realm_2_100m_w_tandem/gnd_1/mosaicing/elevation/gtiff/elevation.tif`: float32 2507 x 1648,
  0.15 m/px, EPSG 32633, top-left corner E 592007.1 / N 5285877.0, NaN nodata, 16 % valid,
  with a ≈200 m outlier tail (filtered below 300 m). Over the forest at the start of the
  tandem window it reads only 25–35 m below a camera that is ~80 m above the nearby fields,
  so treat forest frames as unreliable reference. Only the tandem window overlaps it; the
  `realm_2_with_depth` windows 70–97 lie west of or at its edge (coverage 0–0.1).
- `custom/*.png` are hand-made comparison figures (DepthSplat vs MoGe); MoGe is not in the
  repo.

---

## 3. Pipeline (what exists now)

### 3.1 Scene builder: `scripts/realm_gs_scenes.py` (new, verified)

```
python scripts/realm_gs_scenes.py --dataset realm_2_with_depth --offset 70 --window 7 \
    --stride 7 --num-windows 4 --context-step 2
python scripts/realm_gs_scenes.py --dataset realm_2_100m_w_tandem --window 7 --keep-existing
```

- Windows of consecutive keyframes become scenes `<dataset>_w<first>-<last>` under
  `datasets/custom_images_gs/test/` (default root), with poses in metres relative to the
  window's first context camera (identity asserted, tolerance 1e-4).
- Split: `--context-step k` (interleaved; targets are the other frames) or
  `--context-first n` (extrapolation). The index `assets/evaluation_index_realm_gs.json` gets
  **disjoint** context/target lists; `--index-targets all` is the input-view reprojection
  variant (inflates PSNR).
- `cameras.json` carries `anchor_c2w_utm`, `anchor_frame_id`, `units: "m"`, `utm_epsg`,
  `source_indices`, `context_indices`, `target_indices`, `context_baseline_m`, and per frame
  `index`, `source_index`, `frame_id`, `role`, `depth_reference`. No near/far (they are
  dataset-wide, see 3.2) and no scale.
- Windows with a pose jump above `--max-step` (40 m) or a missing pose are rejected with a
  warning. `--keep-existing` appends to the root and merges the index.

Two window configurations exist (both verified):

| Configuration | Root / index | Window | Context / target | Scenes |
|---|---|---|---|---|
| 7-frame windows | `datasets/custom_images_gs`, `assets/evaluation_index_realm_gs.json` | 7, stride 7 | `[0,2,4,6]` / `[1,3,5]` | 4 x `realm_2_with_depth` (70–97) + 1 tandem (0–6) |
| triplets | `datasets/custom_images_gs3`, `assets/evaluation_index_realm_gs3.json` | 3, stride 3 | `[0,2]` / `[1]` | 9 x `realm_2_with_depth` (70–96) + 4 tandem (0–11) |

```
python scripts/realm_gs_scenes.py --dataset realm_2_with_depth --offset 70 --window 3 --stride 3 \
    --num-windows 9 --context-step 2 --data-root datasets/custom_images_gs3 \
    --index-path assets/evaluation_index_realm_gs3.json
python scripts/realm_gs_scenes.py --dataset realm_2_100m_w_tandem --window 3 --stride 3 --num-windows 4 \
    --context-step 2 --data-root datasets/custom_images_gs3 \
    --index-path assets/evaluation_index_realm_gs3.json --keep-existing
```

### 3.2 Configuration

- `config/dataset/custom_images.yaml`: 480x640, `near 10`, `far 500`. The single source of
  `image_shape`, `near`, `far`; they apply to the whole scene and every view. Change the
  yaml, never a command line.
- `config/experiment/realm.yaml` (new): same `model`/`loss` block as `experiment/re10k.yaml`
  (the checkpoints require it, verified identical after composition) plus
  `trainer.num_nodes`, and **no dataset values**, so the yaml wins. With `+experiment=re10k`
  the experiment's own `near 0.5`, `far 100`, `image_shape [256,256]` are merged last and win
  (verified); deleting `dataset.near/far` from such a command does **not** restore the yaml.
  Hydra parses `dataset.near=40` as an int and the typed config then fails with a
  `UnionMatchError`; write `40.0`.
- `config/model/encoder/depthsplat.yaml` (new keys): `normalize_scene_scale` (default
  false), `scene_scale_mode` (`nearfar` default, or `baseline`), `scene_scale_target_depth`
  (3.0), `scene_scale_baseline` (1.0). See 3.4.
- `config/main.yaml`: `test.save_rendered_depth` (default false).
- `.vscode/launch.json`: the two "Depth Inference" entries use `+experiment=realm`; "Realm
  Triplet Depth" uses `--output-dir`; "GS Eval: realm windows" runs `scripts/realm_gs_eval.py`
  on the 7-frame root (`outputs/gs-eval-realm`), "GS Eval: realm triplets" on the triplet root
  (`outputs/gs-eval-realm-triplets`). Both pass `--gaussian-scale-max 0.1 --consistency`.

### 3.3 Runners

- **GS, in-process:** `src/gs_inference.py` (`gs_inference_overrides()`,
  `run_gs_inference(cfg, full_fp32=True)`) and the CLI `scripts/realm_gs_eval.py`:

  ```
  python scripts/realm_gs_eval.py --output-dir outputs/gs-eval-realm --consistency --dem-eval
  python scripts/realm_gs_eval.py --output-dir outputs/gs-eval-realm-hybrid \
      --depth-checkpoint pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth \
      --no-scene-scale-normalization --gaussian-scale-max 1.0
  ```

  Defaults: `datasets/custom_images_gs`, `assets/evaluation_index_realm_gs.json`, base GS
  checkpoint, ViT-B, `num_scales 2`, `upsample_factor 4`, `lowest_feature_resolution 8`,
  normalisation on (`nearfar`, target 3.0), `gaussian_scale_max 0.1` (normalised units),
  fp32, PLY + colour renders + encoder depth + rendered depth, `num_workers 0`. Writes
  `manifest.json` (settings, yaml near/far, git commit, precision), `summary.md/json`
  (per-scene, per-view PSNR/SSIM/LPIPS recomputed from the saved images), and optionally
  `depth_consistency_encoder.json` and `dem_eval_{encoder,rendered}.json`.
  `--depth-checkpoint` builds the hybrid (GS checkpoint, then the depth-only weights into
  `encoder.depth_predictor`, as `src/main.py` also does in test mode).
- **Depth-only:** unchanged (`scripts/realm_triplet_depth.py`, `src/depth_inference.py`);
  for a windows-root scene use `depth_inference_overrides(dataset_root="datasets/custom_images_gs",
  scene_selection=<scene>, ...)` with near/far read from the yaml (the `all` sampler makes
  every frame a context view).
- **`python -m src.main`** still works for both (`+experiment=realm ...`), at TF32.

### 3.4 Model-side changes (verified)

- `src/model/encoder/encoder_depthsplat.py`: opt-in scene-scale normalisation. When
  `normalize_scene_scale` is on, `forward()` multiplies the context camera positions and
  near/far by a per-example factor, runs the network, and divides depths, Gaussian means
  (and scales; covariances by the square) back before anything leaves `forward()`. The
  decoder, PLY export, rendered depth and saved depth maps therefore stay metric.
  `scene_scale_mode: nearfar` uses `scene_scale_target_depth / sqrt(near·far)` (0.0424 for
  10/500: 100 m → 4.2 units); `baseline` uses `scene_scale_baseline / mean consecutive
  context baseline`. The factor is printed per batch (`[scene scale] factor ...`).
  `gaussian_scale_max` is applied in normalised units when normalisation is on.
- `src/model/model_wrapper.py`: `test.save_rendered_depth` renders the rasteriser z-depth at
  the target views to `images/<scene>/depth_rendered/<idx>.npy` (+ png); the chunked
  rendering path concatenates depth too. `on_test_end` no longer crashes with
  `KeyError: 'save'` (the fork's `save` timer had no skip entry).
- Depth conventions (verified in code): predicted depth, rendered depth and the plane sweep
  are all camera z-depth; unproject with `K⁻¹ [u, v, 1] · z`.

### 3.5 Evaluation tools (new, verified)

| Script | What it does |
|---|---|
| `scripts/gs_scores_table.py <run_dir>` | Per-scene, per-view PSNR/SSIM/LPIPS from `images/<scene>/color/<idx>.png` and `<idx>_gt.png` with the same functions `src.main` uses; writes `summary.md/json`; independent of dataloader order. |
| `scripts/depth_consistency.py --run-dir R --data-root D [--source encoder\|rendered\|reference]` | Cross-view consistency without any reference: unproject view i, project into view j, compare with view j's depth. Reports median relative error, inlier fraction (5 %) and correlation per ordered pair and per scene. A constant depth map gives inliers 1.0 with correlation ≈ 0 (far-plane signature). All ordered pairs are averaged, so wide-baseline pairs with little overlap pull the numbers down (see W9). |
| `scripts/realm_dem_eval.py --run-dir R --data-root D [--source encoder\|rendered]` | Disagreement with the reference DEM in UTM: camera height above the DEM (median within `--nadir-radius` 10 m) vs the predicted depth at the principal point, and per-pixel `point_z − DEM_z` statistics. Requires scenes from `realm_gs_scenes.py` (anchor pose). Reads the GeoTIFF tags with `tifffile`; no rasterio. |
| `scripts/compare_depth_to_reference.py A B [--no-align]` | Scale-aligned (default) or absolute comparison of two depth stacks. |
| `scripts/export_gaussians_utm.py --ply P --scene-dir S` | Re-expresses a run's PLY in UTM with a local origin (sidecar JSON with EPSG, origin, bbox). Verified on the tandem window: bbox E 592000–592293, N 5285558–5285832. |
| `scripts/visualize_depth_npy.py` | Unchanged viewer. |

---

## 4. Results (verified 2026-10-16, full fp32 unless stated)

### 4.1 Far-plane collapse and its cure

Scene: `custom/realm_2` frames 70–76 converted with `convert_realm_to_custom_images.py`
(camera Z ≈ 477–483 m, terrain ~75–100 m below). Context `[0,3,6]`, held-out targets
`[1,2,4,5]` unless noted. "far plane" = every encoder depth pixel at `far`.

| Run | Checkpoint | Context | near/far | Scene units | PSNR (dB) | Encoder depth |
|---|---|---|---|---|---|---|
| smoke test (targets incl. context) | GS base | 4 | 10/500 | metric | 12.95 | far plane |
| `v > 3` path check | depth-only base | 4 | 10/500 | metric | – | medians 109–124 m |
| 3 views | GS base | 3 | 10/500 | metric | 12.84 | far plane |
| tight metric range | GS base | 3 | 40/150 | metric | 17.51 | far plane (150 m) |
| translations ÷ 20 (scratch copy) | GS base | 3 | 0.5/25 | ÷20 | 18.93 (`gsm` 1.0) / 18.85 (`gsm` 0.05) | 6.0–6.8 units = 120–135 m |
| normalisation, `baseline` rule (factor 0.0262) | GS base | 3 | yaml | auto | 19.44 | 99–109 m |
| normalisation, `nearfar` rule (factor 0.0424) | GS base | 3 | yaml | auto | **19.85** | 113–126 m |

Conclusions: the `v > 3` matching path is fine; the GS weights collapse at metric scale
regardless of near/far; room-scale presentation fixes it; the normalisation reproduces the
fix from metric inputs and returns metric outputs (medians agree with the depth-only
checkpoint to ~10 %).

### 4.2 Windows: held-out novel-view scores

Five windows (`realm_2_with_depth` 70–76, 77–83, 84–90, 91–97 and `realm_2_100m_w_tandem`
0–6), context `[0,2,4,6]`, targets `[1,3,5]`, 15 views.

| Variant | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| GS, `baseline` rule, `gsm` 0.05 | 16.35 ± 2.57 | 0.325 | 0.479 |
| GS, `baseline` rule, `gsm` 0.1 | 16.57 ± 2.55 | 0.331 | 0.478 |
| GS, `baseline` rule, `gsm` 0.3 | 16.62 ± 2.67 | 0.341 | 0.480 |
| **GS, `nearfar` rule, `gsm` 0.1 (default)** | **16.87 ± 2.78** | **0.334** | **0.474** |
| Hybrid (GS head + depth-only predictor), metric, `gsm` 1.0 m | 16.02 ± 2.66 | 0.325 | 0.526 |

Per window (`nearfar` / hybrid): tandem 13.75 / 12.67; w70–76 19.70 / 19.01;
w77–83 15.80 / 15.31; w84–90 16.16 / 15.36; w91–97 18.93 / 17.74. The `gaussian_scale_max`
grid is flat (0.3 dB over 0.05–0.3), so 0.1 stays the default. Outputs:
`outputs/gs-eval-realm-nf` (default), `outputs/gs-eval-realm` (baseline rule),
`outputs/gs-eval-realm-gsm{0.05,0.3}`, `outputs/gs-eval-realm-hybrid`.

**PSNR is a weak geometry check on flat terrain:** window 70–76 scores best (19.7 dB) while
its GS depth is wrong (next section), because a wrong plane still reprojects nearly
featureless lawn well.

**Triplet configuration** (13 windows of 3, context `[0,2]`, the middle frame held out,
normalised GS, `nearfar` rule, `gsm` 0.1, fp32; `outputs/gs-eval-realm-triplets`):
**17.86 ± 3.06 dB, SSIM 0.350, LPIPS 0.453** over 13 views, peak GPU memory 2.2 GB. Higher
than the 7-frame windows because the target lies between two context cameras only 7–30 m
apart. Per triplet: `realm_2_with_depth` 70–78 (lawn) 20.6–22.8 dB, 79–87 14.4–14.9 dB,
88–96 18.3–18.9 dB; tandem 0–2 12.2 dB, 3–11 16.0–20.5 dB. Cross-view consistency
(inliers / correlation over the two ordered pairs) is 0.83–0.88 / 0.68–0.95 on the tandem
triplets 6–11 and `realm_2_with_depth` 91–96, 0.3–0.6 / 0.2–0.8 elsewhere. The lawn triplets
70–78 still predict ~140 m medians (camera ~30 m above ground): the content failure of
section 4.3 does not depend on the number of views.

### 4.3 Geometry: cross-view consistency and encoder depth

Inlier fraction (5 % relative) / correlation, mean over ordered context pairs; encoder depth
medians in metres for the context views.

| Window | GS `nearfar` | Hybrid | Depth-only (all 7 frames as context) |
|---|---|---|---|
| tandem 0–6 | 0.17 / 0.06; depth 141, 155, 99, 86 | 0.22 / 0.13; depth 86, 107, 94, 91 | 0.31 / 0.14; nadir depths 79–106 |
| w70–76 (featureless lawn, camera ~30 m above ground) | 0.28 / 0.20; depth 237, 259, 244, 219 | 0.35 / 0.33; depth 38, 34, 37, 36 | 0.22 / 0.67; depth 50–67 |
| w77–83 | 0.10 / 0.02; depth 63, 46, 36, 42 | 0.13 / 0.02; depth 39, 41, 48, 64 | – |
| w84–90 | 0.10 / 0.17; depth 49, 38, 35, 50 | 0.32 / 0.44; depth 49, 37, 37, 41 | – |
| w91–97 | 0.27 / 0.16; depth 36, 40, 52, 56 | 0.48 / 0.31; depth 37, 41, 49, 57 | – |
| realm_2 verification scene (3 context) | 0.49 / 0.40 | – | – |

Reading: the hybrid is more self-consistent in four of five windows and gives plausible
depth on the lawn window where the GS predictor fails at every tested factor (0.042–0.045 →
220–280 m; ≥ 0.085 → far plane). The forest frames at the start of the tandem window are a
second failure of the GS predictor (141–155 m where the depth-only and hybrid predictors
say 83–107 m).

### 4.4 Disagreement with the reference DEM (tandem window only)

Open-field context frames 4 and 6 (DEM coverage 0.73–0.87, the only reliable part):

| Predictor | Nadir depth / camera height above DEM | Median point height − DEM (m) |
|---|---|---|
| depth-only (7 views) | 1.22 / 1.07 | −10.9 / −4.6 |
| GS, `baseline` rule | 1.07 / 1.04 | −1.5 / −4.7 |
| GS, `nearfar` rule (default) | 1.11 / 1.08 | −4.5 / −7.9 |
| hybrid | 1.23 / 1.15 | −10.8 / −13.3 |

Rendered depth at held-out targets 3 and 5: GS `baseline` 1.26 / 1.07, hybrid 1.40 / 1.14.
Per-pixel agreement is poor everywhere (RMS 20–40 m, < 15 % within 3 m): trees, DEM holes and
model errors all contribute and cannot be separated with this reference. Forest frames 0–2
disagree by 40–60 m against a DEM that is itself implausible there. Use these numbers as a
coarse scale check only.

### 4.5 Georeferenced export (verified)

`outputs/gs-eval-realm/gaussians/realm_2_100m_w_tandem_w0000-0006_utm.ply` + `.json` from
`export_gaussians_utm.py`. Opacity is low for most Gaussians (median 0.20; 8 % above 0.5),
so filter by opacity before comparing the splat cloud with the DEM or orthophoto.

---

## 5. Remaining work

- **W6 (P1, L) Whole-trajectory reconstruction.** Not started. Needs a single global anchor
  for all windows (the builder anchors per window), merging/deduplication of per-window
  Gaussians, and memory planning (6.3 GB per 4+3-view window on the 12 GB GPU). The
  small 4–10-view checkpoint is an option for longer windows.
- **W8 (P0, decision) Normalised GS vs hybrid.** Images: normalised GS (+0.85 dB, better
  LPIPS). Geometry: hybrid (metric without normalisation, consistent on hard content). A
  third option is fine-tuning the GS checkpoint's depth predictor on metric aerial data,
  which is a training project. Decide per use case (section 7).
- **W9 (P1, S) Consistency metric.** Report adjacent context pairs (or overlap-weighted
  pairs) in `depth_consistency.py`; the all-pairs mean mixes 27 m and 80 m baselines.
- **W10 (P1, M) Failure cases.** Textureless lawn (w70–76) and forest (tandem 0–2):
  evaluate the hybrid and the depth-only checkpoint with more context views, and consider
  a confidence map (match probability is available in the encoder as `match_prob`).
- **W11 (P2, S) Tandem window image quality** (13.8 dB vs 16–20 elsewhere): check
  exposure/white-balance differences between keyframes and the forest content before
  blaming geometry.
- **W12 (P2, S) MoGe.** If MoGe depths are provided as `.npy`,
  `compare_depth_to_reference.py --no-align` compares them in metres.
- **W7 leftovers (P3).** Optionally switch `scripts/inference_depth_realm.sh`,
  `run_vits_experiment.sh` and `depth_inference_overrides()` to `+experiment=realm`; add a
  hybrid launch entry; delete the diagnostic output directories listed in section 6 when
  no longer needed. `TODO.md`'s ViT-L item is unrelated and still open.

---

## 6. Pitfalls (verified unless stated)

- Hydra override types: `dataset.near=40` is an int and breaks the typed config; use `40.0`.
- `+experiment=re10k` resets `image_shape`/`near`/`far` to the experiment's values; use
  `+experiment=realm` (section 3.2).
- Far-plane signature: every depth pixel at `far`, consistency inliers 1.0 with correlation
  ≈ 0, DEM height difference ≈ −(far − height above ground). Seen with the GS checkpoints at
  metric scale and with the `baseline` normalisation rule on a slow segment (factor 0.09).
- The GS predictor also fails on featureless content at any sane factor (section 4.3); check
  the consistency numbers, not only PSNR.
- `scripts/realm_triplet_depth.py --output-dir` must be absolute; `realm_dem_eval.py` and
  `depth_consistency.py` need `--data-root` (the root whose `cameras.json` has the anchor).
- `realm_gs_scenes.py` wipes `<data-root>/test` unless `--keep-existing` is given; use a
  separate root per experiment type.
- Index target lists that include the context views inflate PSNR
  (`convert_realm_to_custom_images.py` does this; `realm_gs_scenes.py` does not).
- `+experiment=dl3dv` would crop realm images to 448x640 (patch shim 64 px); `realm` and
  `re10k` keep 480x640 (16 px).
- `test.save_depth` in the GS path saves the encoder depth at context views;
  `test.save_rendered_depth` is the rasterised depth at target views.
- The GS PLY is written in the first context camera's frame (identity for these scenes);
  `export_gaussians_utm.py` applies the anchor pose.
- TF32 vs fp32 moves depth by metres; `src.main` is TF32, the in-process runners fp32.
- Generated data produced before the dataset rename keeps the old names
  (`outputs/depthsplat-depth-base-realm_1_with_depth-triplets`,
  `datasets/custom_images_triplets/test/realm_1_with_depth_off0080_*`,
  `outputs/dataset_cfg_with_gt_m.pt2`).
- Diagnostic outputs of 2026-10-16 (gitignored, safe to delete): `outputs/diag_*`,
  `outputs/verify_*`, `outputs/gs-w0070-*`, `outputs/depthsplat-gs-base-realm_2-smoke`,
  `outputs/depthonly-*`; evaluation runs: `outputs/gs-eval-realm*`.

---

## 7. Decisions for the user

1. **Normalised GS or hybrid** as the default for the situation-map use case (section 4.2
   vs 4.3). The runner supports both; the launch entry uses the normalised GS.
2. **`scene_scale_target_depth` = 3.0** (10/500 m → factor 0.0424). Verified working range
   of the GS checkpoints: terrain at ~2.6–6 normalised units; collapse observed at ≥ ~9.
3. **Window protocol:** 7 keyframes with context every second frame (3 held-out targets) or
   triplets with the middle frame held out; both are built and evaluated (sections 3.1 and
   4.2). Longer windows or extrapolation splits (`--context-first`) are untested.
4. Which flight segments matter: the early climb (30–70 m above ground) and the ~80 m
   cruise behave differently, and only the tandem window has reference-DEM coverage.
5. Should the uncommitted `compose.yaml` host-path change be committed?

---

## 8. File inventory (2026-10-16)

New: `config/experiment/realm.yaml`, `src/gs_inference.py`, `scripts/realm_gs_scenes.py`,
`scripts/realm_gs_eval.py`, `scripts/gs_scores_table.py`, `scripts/depth_consistency.py`,
`scripts/realm_dem_eval.py`, `scripts/export_gaussians_utm.py`,
`assets/evaluation_index_realm_gs.json`, `assets/evaluation_index_realm_gs3.json`,
`assets/evaluation_index_realm_diag3.json`, `datasets/custom_images_gs/` and
`datasets/custom_images_gs3/` (generated), this file.

Modified: `src/model/model_wrapper.py` (rendered depth, `on_test_end` fix),
`src/model/encoder/encoder_depthsplat.py` (scene-scale normalisation),
`config/model/encoder/depthsplat.yaml`, `config/main.yaml`,
`scripts/compare_depth_to_reference.py` (`--no-align`), `scripts/depth_export_common.py`
(dataset choices), `scripts/{export_dataset,aot_compile_depth,run_depth_aoti,export_depth_model,visualize_depth_npy}.py`
(dataset rename), `.vscode/launch.json`, `assets/evaluation_index_realm.json` (7-frame
smoke-test index, regenerated by the converter).

Nothing is committed.
