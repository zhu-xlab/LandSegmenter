""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn

from sam2.utils.clip_visual_transformer import VisionTransformer, LayerNorm

vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32, 'embed_dim':512}

# @dataclass
# class CLIPVisionCfg:
#     layers: Union[Tuple[int, int, int, int], int] = 12
#     width: int = 768
#     head_width: int = 64
#     mlp_ratio: float = 4.0
#     patch_size: int = 32
#     image_size: Union[Tuple[int, int], int] = 224
#     # others
#     ls_init_value: Optional[float] = None  # layer scale initial value
#     patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
#     attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
#     attn_pooler_queries: int = 256  # n_queries for attentional pooler
#     attn_pooler_heads: int = 8  # n heads for attentional_pooling
#     no_ln_pre: bool = False  # disable pre transformer LayerNorm
#     pos_embed_type: str = 'learnable'
#     final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
#     pool_type: str = 'tok'
#     output_tokens: bool = False
    

def build_clip_vision_encoder(
        embed_dim: int = 512,
        layers: Union[Tuple[int, int, int, int], int] = 12,
        width: int = 768,
        head_width: int = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 32,
        image_size: Union[Tuple[int, int], int] = 224,
        # others
        ls_init_value: Optional[float] = None,  # layer scale initial value
        patch_dropout: float = 0.,  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
        attentional_pool: bool = False,  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
        attn_pooler_queries: int = 256,  # n_queries for attentional pooler
        attn_pooler_heads: int = 8,  # n heads for attentional_pooling
        no_ln_pre: bool = False,  # disable pre transformer LayerNorm
        pos_embed_type: str = 'learnable',
        final_ln_after_pool: bool = False,  # apply final LayerNorm after pooling
        pool_type: str = 'tok',
        output_tokens: bool = False,
        fix_weights: bool = True,
        load_pretrained: bool | str = False,
):
    # if isinstance(vision_cfg, dict):
    #     vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = nn.GELU

    vision_heads = width // head_width
    norm_layer = LayerNorm

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        width=width,
        layers=layers,
        heads=vision_heads,
        mlp_ratio=mlp_ratio,
        ls_init_value=ls_init_value,
        patch_dropout=patch_dropout,
        attentional_pool=attentional_pool,
        attn_pooler_queries=attn_pooler_queries,
        attn_pooler_heads=attn_pooler_heads,
        pos_embed_type=pos_embed_type,
        no_ln_pre=no_ln_pre,
        final_ln_after_pool=final_ln_after_pool,
        pool_type=pool_type,
        output_tokens=output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    if load_pretrained:
        model = _load_pretrain_weights(model, load_pretrained)
    
    if fix_weights:
        for param in model.parameters():
            param.requires_grad = False

    return model


def _load_pretrain_weights(model, pretrained_path):
    stat_dict = torch.load(pretrained_path)
    # delete text part weights
    for key in list(stat_dict.keys()):
        if not key.startswith('visual.'):
            del stat_dict[key]
    # refine visual part keys
    for key in list(stat_dict.keys()):
        key_ = key[len('visual.'):]
        stat_dict[key_] = stat_dict[key]
        del stat_dict[key]

    # load weights
    model.load_state_dict(stat_dict, strict=True)
    return model