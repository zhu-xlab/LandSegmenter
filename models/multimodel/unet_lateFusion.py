import torch
from typing import Optional, Union, List

import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_encoder
# from segmentation_models_pytorch.base import (
#     SegmentationModel,
#     SegmentationHead,
# )
# from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class Unet_lateF_2branch(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels_1: int = 2,
        in_channels_2: int = 13,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.model1 = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=None,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels_1,
            classes=classes,
            activation=activation,
        )
        
        self.model2 = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels_2,
            classes=classes,
            activation=activation,
        )

        self.name = "umm-lateF-2branch-{}".format(encoder_name)
        
        
    def forward(self, x1, x2):
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        return out1, out2
    
    
    @torch.no_grad()
    def predict(self, x1, x2):
        if self.training:
            self.eval()

        out1, out2 = self.forward(x1, x2)

        return out1, out2


class Unet_lateF_3branch(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels_1: int = 2,
        in_channels_2: int = 13,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.model1 = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=None,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels_1,
            classes=classes,
            activation=activation,
        )
        
        self.model2 = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels_2,
            classes=classes,
            activation=activation,
        )
        
        self.model_com = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=None,
            decoder_use_batchnorm=decoder_use_batchnorm,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels_1+in_channels_2,
            classes=classes,
            activation=activation,
        )
        
        
    def forward(self, x1, x2):
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        out_com = self.model_com(torch.cat([x1, x2], dim=1))

        return out1, out2, out_com
    
    @torch.no_grad()
    def predict(self, x1, x2):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        out1, out2, out_com = self.forward(x1, x2)

        return out1, out2, out_com