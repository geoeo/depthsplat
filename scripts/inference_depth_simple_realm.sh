#!/usr/bin/env bash

# Depth prediction on custom realm_1 or realm_2 datasets using pretrained model
# Set REALM_DATASET to "realm_1" or "realm_2" below

REALM_DATASET=${1:-realm_1}

echo "Running depth inference on custom_images dataset: $REALM_DATASET"

# Read image_shape from config so the experiment override doesn't clobber it.
IMAGE_SHAPE=$(python3 -c "import yaml; s=yaml.safe_load(open('config/dataset/custom_images.yaml'))['image_shape']; print(f'{s[0]},{s[1]}')")
echo "Using image_shape: [$IMAGE_SHAPE]"

CUDA_VISIBLE_DEVICES=0 python -m src.main \
+experiment=re10k \
dataset=custom_images \
dataset.roots=[datasets/custom_images] \
dataset.scene_selection=$REALM_DATASET \
dataset.test_chunk_interval=1 \
dataset/view_sampler=arbitrary \
mode=test \
dataset.image_shape=[$IMAGE_SHAPE] \
dataset.view_sampler.num_context_views=2 \
dataset.view_sampler.num_target_views=10 \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=4 \
model.encoder.lowest_feature_resolution=8 \
model.encoder.monodepth_vit_type=vitb \
train.forward_depth_only=true \
checkpointing.pretrained_depth=pretrained/depthsplat-depth-base-352x640-randview2-8-65a892c5.pth \
test.compute_scores=false \
test.save_depth=true \
test.save_depth_concat_img=true \
test.save_depth_npy=true \
output_dir=outputs/depthsplat-depth-base-$REALM_DATASET



