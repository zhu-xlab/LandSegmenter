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
    load_decoder_weights=True,
    load_aux_decoder_weights=False,
    test_mode=False,
    dofa_unfix_weights=False,
    hydra_overrides_extra=[],
    apply_postprocessing=False,
    device=torch.device("cpu"),
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        # hydra_overrides_extra += [
        #     # dynamically fall back to multi-mask if the single mask is not stable
        #     "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        #     "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        #     "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        # ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    if test_mode:
        _load_checkpoint_test(model, ckpt_path)
        model = model.to(device)
        model.eval()
    else:
        _load_checkpoint(model, ckpt_path, ckpt_path_clip, load_decoder_weights, load_aux_decoder_weights)
        _fix_pretrain_weights(model, dofa_unfix_weights)
    return model


def _load_checkpoint(model, ckpt_path, ckpt_path_clip, load_decoder_weights, load_aux_decoder_weights=False):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        if load_decoder_weights:
            # if load_aux_decoder_weights, copy decoder weights to auxiliary decoder weights
            if load_aux_decoder_weights:
                # print("size of sd before copying: ", len(sd))
                sd_ = sd.copy()
                for k in sd_:
                    if 'sam_mask_decoder' in k:
                        k_aux = k.replace("sam_mask_decoder", "sam_mask_decoder_aux")
                        sd[k_aux] = sd_[k]
                        # print(f'Copying decoder weights to auxiliary decoder weights: {k} -> {k_aux}')
            # assert not model.use_auxiliary_decoder, "Auxiliary decoder is not supported when load_decoder_weights is True"
            missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
            # print(f'sd size: {len(sd)}, missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}')
            if missing_keys:
                # print(missing_keys)
                for mk in missing_keys:
                    de_check_str = 'sam_mask_decoder' if load_aux_decoder_weights else 'sam_mask_decoder.' 
                    if (mk.startswith('image_encoder') and 'prompt_generator' not in mk) \
                          or mk.startswith(de_check_str) \
                            or mk.startswith("no_mem_embed"):
                        if mk.split(".")[1] not in ['existance_prediction_head', 'precise_prediction_head']:
                            raise RuntimeError(f'Unexpected missing key: {mk}')
            print(f"Loaded image encoder and decoder weights from SAM2 checkpoint (load_aux_decoder_weights={load_aux_decoder_weights})")
        else:
            sd_new = sd.copy()
            for k in sd:
                if k.startswith("sam_mask_decoder"):
                    sd_new.pop(k)
            missing_keys, unexpected_keys = model.load_state_dict(sd_new, strict=False)
            missing_keys_contain_decoder = False
            if missing_keys:
                for mk in missing_keys:
                    if (mk.startswith('image_encoder') and 'prompt_generator' not in mk) \
                        or mk.startswith("no_mem_embed"):
                        raise RuntimeError(f'Unexpected missing key: {mk}')
                    if mk.startswith("sam_mask_decoder."):
                        missing_keys_contain_decoder = True
            if not missing_keys_contain_decoder:
                raise RuntimeError('Require not to load decoder weights, but missing keys do not contain decoder')    
            print("Loaded encoder weights from SAM2 checkpoint")                
    else:
        print("No SAM2 checkpoint provided, initializing SAM2 model with random weights")
        
    # load clip model
    if ckpt_path_clip is not None:
        sd = torch.load(ckpt_path_clip, map_location="cpu")
        missing_keys, unexpected_keys = model.sam_prompt_encoder.load_state_dict(sd, strict=False)
        if missing_keys:
            for mk in missing_keys:
                if mk.startswith('sam_prompt_encoder'):
                    raise RuntimeError(f'Unexpected missing key when loading CLIP weights: {mk}')
        print("Loaded prompt encoder weights from CLIP checkpoint")
    else:
        print("No CLIP checkpoint provided, initializing prompt encoder with random weights")
    return


def _load_checkpoint_test(model, ckpt_path):
    pref = 'model.'
    if ckpt_path is not None:
        sd_ = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        sd = sd_.copy()
        for k in sd_:
            sd[k[len(pref):]] = sd_[k]
            sd.pop(k)
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        if missing_keys:
            raise RuntimeError("Found missing keys")
        # if unexpected_keys:
        #     raise RuntimeError("Found unexpected keys")
        print("Loaded checkpoint sucessfully")


def _fix_pretrain_weights(model, dofa_unfix_weights=False):
    # fix encoder weights
    for name, param in model.named_parameters():
        if 'image_encoder' in name and 'prompt_generator' not in name:
            if dofa_unfix_weights:
                if 'dofa' in name: continue
            param.requires_grad = False
        if 'sam_prompt_encoder' in name:
            param.requires_grad = False
    print('Image and prompt encoder weights are fixed!')