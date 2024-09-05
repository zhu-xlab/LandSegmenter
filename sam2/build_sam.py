# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2(
    config_file,
    ckpt_path=None,
    ckpt_path_clip=None,
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path, ckpt_path_clip)
    _fix_pretrain_weights(model)
    return model


def _load_checkpoint(model, ckpt_path, ckpt_path_clip):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        if missing_keys:
            # print(missing_keys)
            for mk in missing_keys:
                if mk.startswith('image_encoder') or mk.startswith("sam_mask_decoder"):
                    raise RuntimeError(f'Unexpected missing key: {mk}')
        # if unexpected_keys:
            # print(unexpected_keys)
    else:
        print("No checkpoint provided, initializing sam2 model with random weights")
        
    # load clip model
    if ckpt_path_clip is not None:
        sd = torch.load(ckpt_path_clip, map_location="cpu")
        missing_keys, unexpected_keys = model.sam_prompt_encoder.load_state_dict(sd, strict=False)
        if missing_keys:
            for mk in missing_keys:
                if mk.startswith('sam_prompt_encoder'):
                    raise RuntimeError(f'Unexpected missing key when loading CLIP weights: {mk}')
            # print(missing_keys)
        # if unexpected_keys:
            # print(unexpected_keys)
        # logging.info("Loaded checkpoint sucessfully")
    return


def _fix_pretrain_weights(model):
    # fix encoder weights
    for name, param in model.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        if 'sam_prompt_encoder' in name:
            param.requires_grad = False
    print('Image and prompt encoder weights are fixed!')