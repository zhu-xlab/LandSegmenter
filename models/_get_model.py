# models
import segmentation_models_pytorch as smp
from models.vit.segmenter import create_segmenter
from models.vit.upernet import create_upernet
import utils_comp.finetune as ft


def get_model(framework, backbone, n_classes, n_channels, pretrain_tag, 
              patch_size=None, im_size=None,                 # for transformer-based models
              weight_path=None, weight_load_tag=None,        # for loading weights from a checkpoint
              enc_finetune_tag='etr'):                       # for encoder finetuning options
    if pretrain_tag in ['imagenet','ssl','swsl']: 
        assert n_channels==3, 'Pretrained encoder weights in smp package only support 3 bands as input!'
    
    # choose framework
    # Transformer-based frameworks
    if framework == 'segmenter':
        assert patch_size is not None, 'Please provide patch size for transformer-based models!'
        assert im_size is not None, 'Please provide image size for transformer-based models!'
        net = create_segmenter(backbone, patch_size, im_size, n_channels, n_classes, pretrain_tag)
        print(f"Construct segmenter model with {backbone} as backbone!")
    elif framework == 'upernet':
        assert patch_size is not None, 'Please provide patch size for transformer-based models!'
        assert im_size is not None, 'Please provide image size for transformer-based models!'
        net = create_upernet(backbone, patch_size, im_size, n_channels, n_classes, pretrain_tag)
        print(f"Construct upernet model with {backbone} as backbone!")
    # CNN-based frameworks
    else:
        if framework == 'unet':
            structure = smp.Unet
        elif framework == 'deeplabv3plus': 
            structure = smp.DeepLabV3Plus
        elif framework == 'deeplabv3':
            structure = smp.DeepLabV3
        elif framework == 'pspnet':
            structure = smp.PSPNet
        elif framework == 'linknet':
            structure = smp.Linknet
        elif framework == 'fpn':
            structure = smp.FPN
        elif framework == 'pan':
            structure = smp.PAN
        elif framework == 'manet':
            structure = smp.MAnet
        elif framework == 'unetplusplus':
            structure = smp.UnetPlusPlus
        else:
            raise ValueError("Please provide correct framework type (segmenter, unet, deeplabv3, deeplabv3plus, or pspnet)!")
        net = structure(encoder_name=backbone,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights=pretrain_tag if pretrain_tag in ['imagenet','ssl','swsl'] else None,  # pre-trained weights for encoder initialization
                        in_channels=n_channels,    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=n_classes,         # model output channels (number of classes in your dataset)
                        )
        print(f"Construct {framework} model with {backbone} as backbone using smp package!")   

    # load weights from a checkpoint file
    if pretrain_tag in ['imagenet','ssl','swsl']: 
        print(f"Pre-trained model is loaded from smp package (type:{pretrain_tag})!")
    elif pretrain_tag != 'rand':
        net = ft.get_model_state_dict_from_saved_models(net, weight_path, weight_load_tag, pretrain_tag)
        print(f"Pre-trained model is loaded from '{weight_path}' (loading type:{weight_load_tag})!")
    else:
        print('No pretrained weights, training will start from scratch with random initializations.')
    net = ft.fix_model_weights(net, enc_finetune_tag)

    return net