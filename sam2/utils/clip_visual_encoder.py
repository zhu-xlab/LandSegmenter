""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn

from sam2.utils.clip_visual_transformer import VisionTransformer, LayerNorm

vision_cfg = {'image_size': 224, 'layers': 12, 'width': 768, 'patch_size': 32, 'embed_dim':512}

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 32
    image_size: Union[Tuple[int, int], int] = 224
    # others
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    

def build_clip_vision_encoder(
        vision_cfg: CLIPVisionCfg,
        embed_dim: int = 512,
        fix_weights: bool = True,
        load_pretrained: bool | str = False,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNorm

    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        attentional_pool=vision_cfg.attentional_pool,
        attn_pooler_queries=vision_cfg.attn_pooler_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        pos_embed_type=vision_cfg.pos_embed_type,
        no_ln_pre=vision_cfg.no_ln_pre,
        final_ln_after_pool=vision_cfg.final_ln_after_pool,
        pool_type=vision_cfg.pool_type,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    if load_pretrained:
        load_pretrain_weights(visual, load_pretrained)
    
    if fix_weights:
        for param in visual.parameters():
            param.requires_grad = False

    return visual


def load_pretrain_weights(model, pretrained_path):
    stat_dict = torch.load(pretrained_path)
    # refine keys
    stat_dict_ = stat_dict.copy()
    for key in stat_dict_:
        if key.startswith('visual.'):
            stat_dict[key[7:]] = stat_dict.pop(key)
        else:
            del stat_dict[key]

    # load weights
    model.load_state_dict(stat_dict, strict=True)