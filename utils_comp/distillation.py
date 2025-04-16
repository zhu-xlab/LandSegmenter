# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:54:12 2023

@author: liu_ch
"""
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torchvision.models as models
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base import SegmentationModel


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class KDStudentUnet(SegmentationModel):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super(KDStudentUnet, self).__init__()

        # Initialize the UNet backbone
        self.base_model = Unet(encoder_name=encoder_name,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                               encoder_weights=encoder_weights,     # randomly initialize student model
                               in_channels=in_channels,             # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                               classes=classes)
        
        # get number of inputs for each block in decoder
        n_dins = [list(block.parameters())[-1].shape[0] for block in self.base_model.decoder.blocks]
        n_dins.pop()
        
        # create 1x1 conv layers
        self.conv1x1_ops = nn.ModuleList([nn.Conv2d(n_din, classes, kernel_size=1)
                                          for n_din in n_dins])
        
        # register hook to intermediate output features for consistency loss calculation
        def register_feat_hooks_smp(net, feat_outs):
            def hook_fn_forward(module, input, output):
                feat_outs.append(output)
            # register hook
            register_list = [net.decoder.blocks[0], net.decoder.blocks[1], 
                             net.decoder.blocks[2], net.decoder.blocks[3]]
            for md in register_list:
                md.register_forward_hook(hook_fn_forward)
            return
        self.feats_de = []
        register_feat_hooks_smp(self.base_model, self.feats_de)

    def forward(self, x):
        self.feats_de.clear()
        outs = self.base_model(x)
        
        if self.training:
            # get dimensionally reduced features by conv1x1
            conv1x1_features = []
            for cv1, feat in zip(self.conv1x1_ops, self.feats_de):
                feat_cv1 = cv1(feat)
                conv1x1_features.append(feat_cv1)
    
            return outs, conv1x1_features
        else:
            return outs
 
    
# consistency loss for decoder
def get_decoder_consist_loss_via_pca(souts, touts, cws_all, n_classes, closs, device):
    # pca
    pca = PCA(n_components=n_classes)
    
    # calculate consistency loss
    cl = torch.tensor(0.0, device=device)
    # get consistency loss for each block
    for i, cws in enumerate(cws_all):
        # get pca features from teacher model
        if device == torch.device('cuda'):
            f1, f2 = touts[4+i].cpu().numpy(), touts[3-i].cpu().numpy()
        else:
            f1, f2 = touts[4+i].numpy(), touts[3-i].numpy()
        tf = np.concatenate([f1,f2],axis=1).transpose(0,2,3,1)
        tpc = pca.fit_transform(tf.reshape(-1,tf.shape[-1]))
        tpc = tpc.reshape(tf.shape[0],tf.shape[1],tf.shape[2],-1).transpose(0,3,1,2)
        tpc = torch.from_numpy(tpc).float().to(device=device)
        cl += cws*closs(souts[i],tpc)
    return cl

def get_decoder_consist_loss_via_pca_perF(souts, touts, cws_all, closs, device):
    # calculate consistency loss
    cl = torch.tensor(0.0, device=device)
    # get consistency loss for each block
    for i, cws in enumerate(cws_all):
        # get pca features from teacher model
        if device == torch.device('cuda'):
            f1, f2 = touts[4+i].cpu().numpy(), touts[3-i].cpu().numpy()
        else:
            f1, f2 = touts[4+i].numpy(), touts[3-i].numpy()
        tf = np.concatenate([f1,f2],axis=1).transpose(0,2,3,1)
        pca = PCA(n_components=souts[i].shape[1])
        tpc = pca.fit_transform(tf.reshape(-1,tf.shape[-1]))
        tpc = tpc.reshape(tf.shape[0],tf.shape[1],tf.shape[2],-1).transpose(0,3,1,2)
        tpc = torch.from_numpy(tpc).float().to(device=device)
        cl += cws*closs(souts[i],tpc)
    return cl

def get_decoder_consist_loss(souts, touts, cws_all, closs, device):
    # calculate consistency loss
    cl = torch.tensor(0.0, device=device)
    # get consistency loss for each block
    for i, cws in enumerate(cws_all):
        cl += cws*closs(souts[i],touts[i])
    return cl
    
    
# define teacher encoder    
def get_teacher_encoder(name, n_channels=3, pretrained=True):
    if pretrained: assert n_channels==3, \
        'Pretrained models in smp package can be only applied to 3-channel input!'
    if name=='resnet18':
        encoder = models.resnet18(weights='DEFAULT' if pretrained else None)
        if n_channels != 3:
            encoder.conv1 = nn.Conv2d(n_channels, 64, 7, stride=2, padding=3, bias=False)
        # remove cls headers
        encoder.avgpool = Identity()
        encoder.fc = Identity()
    else:
        raise(ValueError,'Teacher encoder name is invalid!')
    return encoder


# # # # remove prefix generated by hpc # # # # 
def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd


# load the single teacher encoder 
def load_and_fix_single_pretrain_encoder(Tencoder, ckp_path, device=torch.device('cpu'), pretrain_type='ns'):
    # load weights
    if pretrain_type=='dino':
        state_dict = torch.load(ckp_path, map_location=device)['teacher']
        # dino model encoder prefix
        pf_sdict='module.backbone.'
    elif pretrain_type=='ns':
        state_dict = torch.load(ckp_path, map_location=device)
        state_dict = modify_module_state_dict_keys(state_dict['model_state_dict'])
        # smp model encoder prefix
        pf_sdict='encoder.'
        
    # remove the encoder prefix of state_dict
    for k in list(state_dict.keys()):
        if k.startswith(pf_sdict):
            newk = k[len(pf_sdict):]
            state_dict[newk] = state_dict[k]
        del state_dict[k]
    
    #load weights
    Tencoder.load_state_dict(state_dict,strict=False)
    
    # fix 
    for name, param in Tencoder.named_parameters():
        param.requires_grad = False
            
    return


# # # # register hook to intermediate output features for consistency loss calculation # # # # 
def _feat_hook(net, feat_outs, register_list=[], model_type='student'):
    if len(register_list)==0:
        register_list = [net.layer1,net.layer2,net.layer3,net.layer4]
    # def hook_fn func
    if model_type=='student':
        def hook_fn_forward(module, input, output):
            if module.training:
                feat_outs.append(output)
    else:
        def hook_fn_forward(module, input, output):
            feat_outs.append(output)
    # register hook
    for md in register_list:
        md.register_forward_hook(hook_fn_forward)
    return

# for encoder
def register_feat_hooks_encoder(net, feat_outs, model_type='student'):
    assert model_type in ['student', 'teacher'], "Please provide correct model type for hook registration!"
    register_list = [net.layer1,net.layer2,net.layer3,net.layer4]
    _feat_hook(net, feat_outs, register_list=register_list, model_type=model_type)
    return

# for smp models
def register_feat_hooks_smp(net, feat_outs, model_type='student'):
    assert model_type in ['student', 'teacher'], "Please provide correct model type for hook registration!"
    register_list = [net.encoder.relu, net.encoder.layer1, net.encoder.layer2, net.encoder.layer3,
                     net.decoder.blocks[0], net.decoder.blocks[1], net.decoder.blocks[2], net.decoder.blocks[3]]
    _feat_hook(net, feat_outs, register_list=register_list, model_type=model_type)
    return

# for smp encoders
def register_feat_hooks_smp_encoder(net, feat_outs, model_type='student'):
    assert model_type in ['student', 'teacher'], "Please provide correct model type for hook registration!"
    register_list = [net.encoder.layer1, net.encoder.layer2, net.encoder.layer3, net.encoder.layer4]
    _feat_hook(net, feat_outs, register_list=register_list, model_type=model_type)
    return

# for smp decoders
def register_feat_hooks_smp_decoder(net, feat_outs, model_type='student'):
    assert model_type in ['student', 'teacher'], "Please provide correct model type for hook registration!"
    register_list = [net.decoder.blocks[0], net.decoder.blocks[1], net.decoder.blocks[2], net.decoder.blocks[3]]
    _feat_hook(net, feat_outs, register_list=register_list, model_type=model_type)
    return


# # # # lr adaptor # # # #
# ramp down funcs
def sigmoid_rampdown(current, rampup_length):
    """Exponential rampdown modified based on https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return 1 - float(np.exp(-5.0 * phase * phase))


def linear_rampdown(current, rampup_length):
    """Linear rampdown"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 1 - current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return 1 - float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


# ramp up funcs
def sigmoid_rampup(current, rampup_length):
    """Exponential rampdown modified based on https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampdown"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampup(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


