# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.amp import autocast

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP
from sam2.modeling.backbones.image_encoder_adaptor import PromptGenerator


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dims=[144, 288, 576, 1152],
        embed_dim: int = 144,# initial embed dim
        num_heads: int = 2,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 6, 36, 4),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (7, 7),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            16,
            8
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            23,
            33,
            43
        ),
        return_interm_layers=True,  # return feats from every stage
        # adaptor settings
        adaptor: str = None, # if None, no adaptor is used, the following settings are ignored
        scale_factor: int = 32,
        input_type: str = 'fft',       
        prompt_type: str = 'highpass',   # prompt type for fft
        tuning_stage: str = "1234",
        freq_nums: float = 0.25,
        handcrafted_tune: bool = True,
        embedding_tune: bool = True,
        spectralfeat_tune: bool = True,
        use_statistical_feats: bool = True,
        dofa_name: str = 'vit_base_patch16_224',
        dofa_ckpt_path: str = None,
        dofa_unfix_weights: bool = False,
        dofa_wo_rgb: bool = False,  # whether to exclude RGB channels in dofa feature extraction
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # predefined image sizes
        self.sam2_img_size = 1024
        self.dofa_img_size = 224
        self.default_img_size = 256

        # prepare adaptor
        self.embed_dim = embed_dims
        self.stages = stages
        self.adaptor = adaptor if adaptor != 'None' else None
        if self.adaptor is not None:
            self.adaptor_stage_starts = [0] + \
                [sum(stages[:i])+1 for i in range(1,len(stages))]
            self.scale_factor = scale_factor
            self.prompt_type = prompt_type
            self.tuning_stage = tuning_stage
            self.input_type = input_type
            self.freq_nums = freq_nums
            self.handcrafted_tune = handcrafted_tune
            self.embedding_tune = embedding_tune
            self.spectralfeat_tune = spectralfeat_tune
            self.use_statistical_feats = use_statistical_feats
            self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dim,
                                                    self.tuning_stage, self.stages,
                                                    self.input_type, self.freq_nums, self.adaptor,
                                                    self.default_img_size, self.use_statistical_feats,
                                                    self.handcrafted_tune, self.embedding_tune, 
                                                    self.spectralfeat_tune, dofa_name, dofa_ckpt_path, dofa_unfix_weights,
                                                    )
            self.wo_rgb_in_dofa = dofa_wo_rgb

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed
    
    def forward_without_adaptor(self, x: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            x = x[:, :3] # sam2 only uses RGB channels
            if x.shape[-2:] != (self.sam2_img_size, self.sam2_img_size):
                x = F.interpolate(x, size=(self.sam2_img_size, self.sam2_img_size), mode='bilinear', align_corners=False).contiguous()
            x = self.patch_embed(x)
            # x: (B, H, W, C)

            # Add pos embed
            x = x + self._get_pos_embed(x.shape[1:3])

            outputs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if (i == self.stage_ends[-1]) or (
                    i in self.stage_ends and self.return_interm_layers
                ):
                    feats = x.permute(0, 3, 1, 2)
                    outputs.append(feats)

        return outputs

    def forward_with_adaptor(self, x: torch.Tensor, wave_list: List[float] = None) -> List[torch.Tensor]:
        if self.spectralfeat_tune:
            inp = x
            x = x[:, :3] # sam2 only uses RGB channels
        else:
            x = x[:, :3] # sam2 only uses RGB channels
            inp = x
        if x.shape[-2:] != (self.sam2_img_size, self.sam2_img_size):
            x = F.interpolate(x[:, :3], size=(self.sam2_img_size, self.sam2_img_size), mode='bilinear', align_corners=False).contiguous()
        
        # original sam2 part
        with torch.no_grad():
            x = self.patch_embed(x)   # x: (B, H, W, C)
            # Add pos embed
            x = x + self._get_pos_embed(x.shape[1:3])

        # extract handcrafted features - return dict with key of the stage where the handcrafted features are used
        handcraft_feats = self.prompt_generator.init_handcrafted(inp) 
        # handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(inp)

        # dofa spectral features
        if self.spectralfeat_tune:
            if self.wo_rgb_in_dofa:
                if inp.shape[1] > 3:
                    inp_dofa = inp[:, 3:]
                    if inp_dofa.shape[-1] != self.dofa_img_size:
                        inp_dofa = F.interpolate(inp_dofa, size=(self.dofa_img_size, self.dofa_img_size), mode='bilinear', align_corners=False).contiguous()
                    spectral_feats = self.prompt_generator.init_spectral_feature(inp_dofa, wave_list[3:])
                else:
                    spectral_feats = 0
            else:
                if inp.shape[-1] != self.dofa_img_size:
                    inp = F.interpolate(inp, size=(self.dofa_img_size, self.dofa_img_size), mode='bilinear', align_corners=False).contiguous()
                spectral_feats = self.prompt_generator.init_spectral_feature(inp, wave_list)

        outputs = []
        cur_stage = 0
        for i, blk in enumerate(self.blocks):
            if i in self.adaptor_stage_starts:
                cur_stage += 1
                if str(cur_stage) in self.tuning_stage:
                    spectral_feat_ = None if not self.spectralfeat_tune else 0 if spectral_feats==0 else spectral_feats[str(cur_stage)]
                    prompts = self.prompt_generator.init_prompt(
                        x, handcraft_feats[str(cur_stage)], cur_stage,
                        spectral_feature=spectral_feat_,
                    )   # 0: handcrafted, 1: reprojected embeddings, 2: spectral features
                cur_depth = 0
            # integrate adaptor prompt features
            if str(cur_stage) in self.tuning_stage:
                x = self.prompt_generator.get_prompt(
                    x, prompts, cur_stage, cur_depth
                )
            # sam2 forword path
            x = blk(x)
            cur_depth += 1
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs

    def forward(self, x: torch.Tensor, wave_list: List[float] = None) -> List[torch.Tensor]:
        # with autocast(device_type=x.device.type):
        if self.adaptor is not None:
            return self.forward_with_adaptor(x, wave_list)
        else:
            return self.forward_without_adaptor(x)