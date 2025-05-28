import torch
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder
from segmentation_models_pytorch.decoders.pspnet.decoder import PSPDecoder


# decoder dictionary
decoders = {'unet': UnetDecoder, 'deeplabv3': DeepLabV3Decoder, 'deeplabv3plus': DeepLabV3PlusDecoder, 'pspnet': PSPDecoder}


def get_decoder(encoder_name, decoder_name, encoder_channels, decoder_use_batchnorm=True):
    if decoder_name not in decoders:
        raise ValueError('Not supported decoder {}'.format(decoder_name))
    if decoder_name == 'unet':
        decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
        )
        out_channels = 16
    elif decoder_name == 'deeplabv3':
        decoder = DeepLabV3Decoder(
            encoder_channels=encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
        )
        out_channels = 256
    elif decoder_name == 'deeplabv3plus':
        decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )
        out_channels = 256
    elif decoder_name == 'pspnet':
        decoder = PSPDecoder(
            encoder_channels=encoder_channels,
            out_channels=512,
            use_batchnorm=decoder_use_batchnorm,
            dropout=0.2,
        )
        out_channels = 512
    return decoder, out_channels


class BridgeBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        return out

class ft_model_s12(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_name: str = "unet",   
        decoder_use_batchnorm: bool = True,
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
        
        self.encoder2 = get_encoder(
            encoder_name,
            in_channels=in_channels_2,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        self.bridge_blocks = []
        for n_feats in self.encoder1.out_channels[1:]:
            self.bridge_blocks.append(BridgeBlock(n_feats*2, n_feats))

        self.decoder, out_channels = get_decoder(encoder_name, decoder_name, 
                                                 encoder_channels=self.encoder1.out_channels, 
                                                 decoder_use_batchnorm=decoder_use_batchnorm)

        self.segmentation_head = SegmentationHead(
            in_channels=out_channels,
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
        
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        
        x1, x2 = x[:,:self.in_channels_1], x[:,self.in_channels_1:]

        self.check_input_shape(x1)

        features1 = self.encoder1(x1)
        features2 = self.encoder2(x2)
        features = [torch.cat([features1[0], features2[0]], dim=1)]
        for f1, f2, bridge_block in zip(features1[1:], features2[1:], self.bridge_blocks):
            feature = bridge_block(torch.cat([f1, f2], dim=1))
            features.append(feature)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks
    
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

        masks = self.forward(x1, x2)

        return masks