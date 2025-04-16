'''
Codes were adapted from: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/models/models.py#L160
Settings were adapted from: https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/models/upernet_vit-b16_ln_mln.py
'''
import timm
import torch
import torch.nn as nn

from einops import rearrange

from models.vit._decoders import UPerHead
from models.vit._encoders import create_encoder


class UperNet(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        im_size=224,
        patch_size=16,
        out_indices=(2, 5, 8, 11),
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = encoder.embed_dim
        self.out_indices = out_indices
        self.resize_scale_factors = [patch_size, patch_size // 2, patch_size // 4, patch_size // 8]
        self.hn_patch = im_size // self.patch_size

    def get_middle_features(self, x, wave_list=None):
        # Dictionary to store outputs from specific layers
        outputs = {}

        # Define hook function to capture the output of a layer
        def get_hook(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook

        # Register hooks for the 3rd, 6th, 9th, and 12th blocks for vit-base/small 
        # or 6th, 12th, 18th, and 24th layers for vit-large
        for oi in self.out_indices:
            self.encoder.blocks[oi].register_forward_hook(get_hook(f'block_{oi}'))
        
        if wave_list is not None:
            _ = self.encoder(x, wave_list=wave_list)
        else:
            _ = self.encoder(x)

        return outputs
    
    def rearrange_middle_features(self, outputs):
        new_outs = []
        for k, rsc in zip(outputs.keys(), self.resize_scale_factors):
            out_ = outputs[k][:,1:,:]
            if out_.shape[1]>self.hn_patch**2:  # to reduce satMAE's channel groups
                assert out_.shape[1] % self.hn_patch**2 == 0, f"Number of channels ({out_.shape[1]}) should be divisible by {self.hn_patch**2}"
                n_groups = out_.shape[1] // self.hn_patch**2
                out_ = rearrange(out_, "b (g h w) c -> b c g h w", g=n_groups, h=self.hn_patch)
                out_ = out_.mean(dim=2)
            else:
                out_ = rearrange(out_, "b (h w) c -> b c h w", h=int(self.hn_patch))
            out_ = nn.functional.interpolate(out_.contiguous(), scale_factor=rsc, mode='bilinear', align_corners=False)  # contiguous(): to avoid RuntimeError: upsample_bilinear2d_nhwc only supports output tensors with less than INT_MAX elements
            new_outs.append(out_)
            # print(k, rsc, out_.shape)
        return new_outs

    def forward(self, im, wave_list=None):
        outputs = self.get_middle_features(im, wave_list=wave_list)
        outputs = self.rearrange_middle_features(outputs)
        masks = self.decoder(outputs)

        return masks


# # # # # functions to create the model # # # # #
def create_upernet(enc_name, patch_size, im_size, in_chans, n_cls, pretrain_tag):
    # create encoder
    encoder = create_encoder(enc_name, im_size, in_chans=in_chans, patch_size=patch_size, pretrain_tag=pretrain_tag)

    # create decoder
    decoder = create_decoder(encoder.embed_dim, n_cls=n_cls)

    # create model
    # set the number of output indices for the decoder
    out_indices = (5, 11, 17, 23) if 'large' in enc_name else (2, 5, 8, 11)
    # create the model
    model = UperNet(encoder, decoder, n_cls=n_cls, im_size=im_size, patch_size=patch_size,
                    out_indices=out_indices)

    return model


def create_decoder(dim, n_cls=10):
    decoder = UPerHead(num_class=n_cls, fpn_inplanes=(dim, dim, dim, dim), 
                       pool_scales=(1, 2, 3, 6), fpn_dim=256)
    return decoder


