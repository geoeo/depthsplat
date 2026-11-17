#!/usr/bin/env bash
# Run the fp32 dataset export, AOT compile, and package inference without debugpy.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="${PYTHON:-/opt/conda/envs/depthsplat/bin/python}"
dataset_pt2="$repo_root/outputs/dataset_cfg_with_reference_100m_s.pt2"
export CUDA_VISIBLE_DEVICES=0

echo "[1/3] Export Dataset (.pt2)"
"$python" "$repo_root/scripts/export_dataset.py" \
    --dataset realm_2_100m_w_tandem \
    --offset 0 \
    --count 3 \
    --output "$dataset_pt2"

echo "[2/3] AOT Compile Depth: fp32 (default)"
"$python" "$repo_root/scripts/aot_compile_depth.py" \
    --dataset-pt2 "$dataset_pt2" \
    --fp32 on \
    --clean

echo "[3/3] Run Depth Package: fp32 (default)"
"$python" "$repo_root/scripts/run_depth_aoti.py" \
    --package "$repo_root/outputs/aoti/fp32/depth_predictor_fp32.pt2" \
    --dataset-pt2 "$dataset_pt2" \
    --benchmark --iters 8 \
    --compare-eager \
    --save-cam-images \
    --output-dir "$repo_root/outputs/so_depths"