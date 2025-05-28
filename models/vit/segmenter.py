import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit._utils import padding, unpadding
from models.vit._decoders import MaskTransformer
from models.vit._encoders import create_encoder


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = decoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im, wave_list=None):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        if wave_list is not None:
            x = self.encoder(im, wave_list=wave_list) 
        else:
            x = self.encoder(im)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 # + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
    


# # # # # functions to create the model # # # # #
def create_segmenter(enc_name, patch_size, im_size, in_chans, n_cls, pretrain_tag):
    encoder = create_encoder(enc_name, im_size, in_chans=in_chans, patch_size=patch_size, pretrain_tag=pretrain_tag)
    decoder = create_decoder(encoder.embed_dim, patch_size=patch_size, n_cls=n_cls)
    model = Segmenter(encoder, decoder, n_cls)
    return model


def create_decoder(dim, n_cls=10, patch_size=16):
    decoder_dict = {"d_encoder": dim, "n_cls": n_cls, "patch_size": patch_size}
    decoder_dict['drop_path_rate'] = 0.0
    decoder_dict['dropout'] = 0.1
    decoder_dict['n_layers'] = 2
    decoder_dict["n_heads"] = dim // 64
    decoder_dict["d_model"] = dim
    decoder_dict["d_ff"] = 4 * dim
    decoder = MaskTransformer(**decoder_dict)
    return decoder
