#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URLs for the checkpoints
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_l_url="${BASE_URL}sam2_hiera_large.pt"


# Download each of the four checkpoints using wget
echo "Downloading sam2_hiera_large.pt checkpoint..."
wget $sam2_hiera_l_url || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

# download dofa checkpoints
wget https://huggingface.co/XShadow/DOFA/resolve/main/DOFA_ViT_base_e120.pth

# download RemoteCLIP checkpoints
wget https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt

echo "All checkpoints are downloaded successfully."
