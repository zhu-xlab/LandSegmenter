#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Download SAM2 checkpoint
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_l_url="${BASE_URL}sam2_hiera_large.pt"
echo "Downloading sam2_hiera_large.pt checkpoint..."
wget $sam2_hiera_l_url || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

# download dofa checkpoint
wget https://huggingface.co/earthflow/DOFA/resolve/main/DOFA_ViT_base_e100.pth

# download RemoteCLIP checkpoint
wget https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

# download LandSegmenter checkpoint
wget https://huggingface.co/vikki23/LandSegmenter/resolve/main/LandSegmenter-epoch%3D49.ckpt

echo "All checkpoints are downloaded successfully."
