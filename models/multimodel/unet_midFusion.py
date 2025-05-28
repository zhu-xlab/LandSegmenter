import torch
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class Unet_midF_2branch(SegmentationModel):
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

        self.encoder1 = get_encoder(
            encoder_name,
            in_channels=in_channels_1,
            depth=encoder_depth,
            weights=None,
        )
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels_2,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder1.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        # if aux_params is not None:
        #     self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        # else:
        #     self.classification_head = None
        self.classification_head = None

        self.name = "umm-midF-2branch-{}".format(encoder_name)
        self.initialize()
        
        
    def forward(self, x1, x2):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x1)

        features1 = self.encoder1(x1)
        features2 = self.encoder(x2)
        decoder_output1 = self.decoder(*features1)
        decoder_output2 = self.decoder(*features2)

        masks1 = self.segmentation_head(decoder_output1)
        masks2 = self.segmentation_head(decoder_output2)

        return masks1, masks2
    
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

        masks1, masks2 = self.forward(x1, x2)

        return masks1, masks2


class Unet_midF_3branch(SegmentationModel):
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

        self.encoder1 = get_encoder(
            encoder_name,
            in_channels=in_channels_1,
            depth=encoder_depth,
            weights=None,
        )
        
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels_2,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.encoder_com = get_encoder(
            encoder_name,
            in_channels=in_channels_1+in_channels_2,
            depth=encoder_depth,
            weights=None,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder1.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.classification_head = None

        self.name = "umm-midF-3branch-{}".format(encoder_name)
        self.initialize()
        
        
    def forward(self, x1, x2):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x1)
        self.check_input_shape(x2)

        features1 = self.encoder1(x1)
        features2 = self.encoder(x2)
        features_com = self.encoder_com(torch.cat([x1, x2], dim=1)) 
        decoder_output1 = self.decoder(*features1)
        decoder_output2 = self.decoder(*features2)
        decoder_output_com = self.decoder(*features_com)

        masks1 = self.segmentation_head(decoder_output1)
        masks2 = self.segmentation_head(decoder_output2)
        masks_com = self.segmentation_head(decoder_output_com)

        return masks1, masks2, masks_com
    
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

        masks1, masks2, masks_com = self.forward(x1, x2)

        return masks1, masks2, masks_com