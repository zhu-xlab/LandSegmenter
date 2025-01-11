# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import torch
import torch.distributed
# import torch.nn.functional as F

from torch import nn
from torch.nn.init import trunc_normal_

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoderCLIP
from sam2.modeling.sam.transformer import TwoWayTransformer, TwoWayTransformer_opt2
from sam2.modeling.sam2_utils import LayerNorm2d, get_logvar_smooth_layer
# from sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0
NUM_MULTIOUTPUTS = 3

class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        use_auxiliary_decoder=False,
        use_logvar_smooth=False,
        fix_logvar_smooth=False,
        image_size=512,
        backbone_stride=16,  # stride of the image backbone output
        use_high_res_features_in_sam=False,
        iou_prediction_use_sigmoid=False,
        noise_mask_in_chans=16,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
        decoder_expand_dim: bool = False,
        decoder_use_opt_two: bool = False,
        decoder_use_feature_norm: bool = False,
    ):
        super().__init__()

        # image backbone
        self.image_encoder = image_encoder
        # auxiliary decoder for co-teaching
        self.use_auxiliary_decoder = use_auxiliary_decoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.hidden_dim = self.image_encoder.neck.d_model
        self.noise_mask_in_chans = noise_mask_in_chans
        # a single token to indicate no memory embedding for image processing
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args

        self._build_sam_heads(decoder_expand_dim, decoder_use_opt_two, decoder_use_feature_norm)
        
        # Model compilation
        if compile_image_encoder:
            # Compile the forward function (not the full module) to allow loading checkpoints.
            print(
                "Image encoder compilation is enabled. First forward pass will be slow."
            )
            self.image_encoder.forward = torch.compile(
                self.image_encoder.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        
        # logvar smooth layer
        self.use_logvar_smooth = use_logvar_smooth
        if self.use_logvar_smooth:
            self.logvar_smoother = get_logvar_smooth_layer(kernel_size=3, n_depth=2, no_grads=fix_logvar_smooth)         


    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Please use the corresponding methods in SAM2VideoPredictor for inference."
            "See notebooks/video_predictor_example.ipynb for an example."
        )


    def _build_sam_heads(self, decoder_expand_dim, decoder_use_opt_two, decoder_use_feature_norm):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        clip_embed_dim = 512 # 512 by default for CLIP (base) / 768 - large
        self.sam_prompt_encoder = PromptEncoderCLIP(
            clip_embed_dim=clip_embed_dim,   # 512 by default for CLIP (base) / 768 - large 
            context_length=77,    # 77 by default for CLIP
            vocab_size=49408,     # 49408 by default for CLIP
            transformer_heads=8,  # 8 by default for CLIP
            transformer_layers=12,  # 12 by default for CLIP
            transformer_width=512,  # 512 by default for CLIP (base) / 768 - large
        )
        self.sam_prompt_fn = torch.nn.Linear(clip_embed_dim, self.sam_prompt_embed_dim, bias=True)
        trans = TwoWayTransformer_opt2 if decoder_use_opt_two else TwoWayTransformer
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=NUM_MULTIOUTPUTS,
            transformer=trans(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            expand_dim=decoder_expand_dim,
            use_opt_two=decoder_use_opt_two,
            use_feature_norm=decoder_use_feature_norm,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        # for noise mask quality prediction
        self.sam_noise_mask_downscaling = nn.Sequential(
            nn.Conv2d(1, self.noise_mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.noise_mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(self.noise_mask_in_chans // 4, self.noise_mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(self.noise_mask_in_chans),
            nn.GELU(),
            nn.Conv2d(self.noise_mask_in_chans, self.hidden_dim, kernel_size=1),
        )
        self.no_noise_mask_embed = nn.Embedding(1, self.hidden_dim)

        # for co-teaching
        if self.use_auxiliary_decoder==1:
            self.sam_mask_decoder_aux = MaskDecoder(
                num_multimask_outputs=NUM_MULTIOUTPUTS,
                transformer=trans(
                    depth=2,
                    embedding_dim=self.sam_prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=self.sam_prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                use_high_res_features=self.use_high_res_features_in_sam,
                iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
                expand_dim=decoder_expand_dim,
                use_opt_two=decoder_use_opt_two,
                use_feature_norm=decoder_use_feature_norm,
                **(self.sam_mask_decoder_extra_args or {}),
            )
            # for noise mask quality prediction
            self.sam_noise_mask_downscaling_aux = nn.Sequential(
                nn.Conv2d(1, self.noise_mask_in_chans // 4, kernel_size=2, stride=2),
                LayerNorm2d(self.noise_mask_in_chans // 4),
                nn.GELU(),
                nn.Conv2d(self.noise_mask_in_chans // 4, self.noise_mask_in_chans, kernel_size=2, stride=2),
                LayerNorm2d(self.noise_mask_in_chans),
                nn.GELU(),
                nn.Conv2d(self.noise_mask_in_chans, self.hidden_dim, kernel_size=1),
            )
            self.no_noise_mask_embed_aux = nn.Embedding(1, self.hidden_dim)
        elif self.use_auxiliary_decoder>1:
            for i in range(self.use_auxiliary_decoder):
                sam_mask_decoder_aux = MaskDecoder(
                    num_multimask_outputs=NUM_MULTIOUTPUTS,
                    transformer=trans(
                        depth=2,
                        embedding_dim=self.sam_prompt_embed_dim,
                        mlp_dim=2048,
                        num_heads=8,
                    ),
                    transformer_dim=self.sam_prompt_embed_dim,
                    iou_head_depth=3,
                    iou_head_hidden_dim=256,
                    use_high_res_features=self.use_high_res_features_in_sam,
                    iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
                    expand_dim=decoder_expand_dim,
                    use_opt_two=decoder_use_opt_two,
                    use_feature_norm=decoder_use_feature_norm,
                    **(self.sam_mask_decoder_extra_args or {}),
                )
                setattr(self, f'sam_mask_decoder_aux_{i}', sam_mask_decoder_aux)
                # for noise mask quality prediction
                sam_noise_mask_downscaling_aux = nn.Sequential(
                    nn.Conv2d(1, self.noise_mask_in_chans // 4, kernel_size=2, stride=2),
                    LayerNorm2d(self.noise_mask_in_chans // 4),
                    nn.GELU(),
                    nn.Conv2d(self.noise_mask_in_chans // 4, self.noise_mask_in_chans, kernel_size=2, stride=2),
                    LayerNorm2d(self.noise_mask_in_chans),
                    nn.GELU(),
                    nn.Conv2d(self.noise_mask_in_chans, self.hidden_dim, kernel_size=1),
                )
                setattr(self, f'sam_noise_mask_downscaling_aux_{i}', sam_noise_mask_downscaling_aux)
                no_noise_mask_embed_aux = nn.Embedding(1, self.hidden_dim)
                setattr(self, f'no_noise_mask_embed_aux_{i}', no_noise_mask_embed_aux)
                

    def forward_image(self, img_batch: torch.Tensor, waves: List[float]):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(img_batch, waves)
        # precompute projected level 0 and level 1 features in SAM decoder
        # with the main decoder
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn_de"] = []
            # # using the main decoder
            backbone_out["backbone_fpn_de"].append(
                self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            )
            backbone_out["backbone_fpn_de"].append(
                self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
            )
            backbone_out["backbone_fpn_de"].append(
                backbone_out["backbone_fpn"][2].clone()
            )
        return backbone_out
    

    def _get_backbone_features_from_aux_decoder(self, backbone_out, aux_decoder_index=True):
        if self.use_high_res_features_in_sam:
            if aux_decoder_index is True:
                decoder = self.sam_mask_decoder_aux
            else:
                assert type(aux_decoder_index) is int, "aux_decoder_index must be an integer or True."
                decoder = getattr(self, f'sam_mask_decoder_aux_{aux_decoder_index}')
            
            backbone_out_aux = []
            backbone_out_aux.append(
                decoder.conv_s0(backbone_out["backbone_fpn"][0])
            )
            backbone_out_aux.append(
                decoder.conv_s1(backbone_out["backbone_fpn"][1])
            )
            backbone_out_aux.append(
                backbone_out["backbone_fpn"][2]
            )
        return backbone_out_aux


    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        # backbone_out = backbone_out.copy()
        # assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out) >= self.num_feature_levels, f"backbone_out must have at least {self.num_feature_levels} levels, actual: {len(backbone_out)}"

        feature_maps = backbone_out[-self.num_feature_levels :]
        # vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        # feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        # vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        # return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
        return vision_feats