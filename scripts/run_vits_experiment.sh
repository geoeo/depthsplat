#!/usr/bin/env bash
# Run vits small-model depth inference on a realm scene with configurable near/far,
# then compare (up-to-scale) against the base-model reference depth maps.
set -e

SCENE=${1:-realm_1}
NEAR=$(printf "%.4f" "${2:-10.0}")
FAR=$(printf "%.4f" "${3:-1000.0}")
TAG=${4:-default}

OUTPUT_DIR=outputs/exp_vits_${SCENE}_${TAG}
rm -rf "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
dataset=custom_images dataset.roots=[datasets/custom_images] dataset.scene_selection=$SCENE \
dataset.test_chunk_interval=1 dataset/view_sampler=realm mode=test \
dataset.image_shape=[480,640] dataset.near=$NEAR dataset.far=$FAR \
model.encoder.num_scales=1 model.encoder.upsample_factor=8 model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vits train.forward_depth_only=true model.encoder.train_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-small-352x640-randview2-8-e807bd82.pth \
test.compute_scores=false test.save_depth=true test.save_depth_concat_img=false test.save_depth_npy=true \
output_dir=$OUTPUT_DIR 2>&1 | grep -E "encoder:|Error" || true

echo "--- near=$NEAR far=$FAR scene=$SCENE tag=$TAG ---"
/opt/conda/envs/depthsplat/bin/python scripts/compare_depth_to_reference.py \
  "$OUTPUT_DIR/images/$SCENE/depth" \
  "outputs/depthsplat-depth-base-$SCENE/images/$SCENE/depth"
