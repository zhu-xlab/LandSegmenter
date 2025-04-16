import timm
import torch
import models.vit._satmae as satmae
import models.vit._satmae_3b as satmae_3b
import models.vit._dofa as dofa

# # sentinel-2 band info
# BANDS_3 = [3,2,1]
# BANDS_9 = [1,2,3,4,5,6,7,11,12]
# BANDS_10 = [1,2,3,4,5,6,7,8,11,12]
# BANDS_12 = [0,1,2,3,4,5,6,7,8,9,11,12] # remove B10 - cirrus band
# WAVELENS = [0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.375, 1.61, 2.19]

def create_encoder(name, im_size, in_chans=3, patch_size=16, pretrain_tag=None):
    if 'satmae' in pretrain_tag:
        if in_chans==3:
            encoder = satmae_3b.__dict__[name](img_size=im_size, 
                                               patch_size=patch_size, 
                                               in_chans=in_chans,
                                               num_classes=-1,
                                               global_pool=False
                                               )
        else:  # 10b
            grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
            encoder = satmae.__dict__[name](img_size=im_size, 
                                            patch_size=patch_size, 
                                            in_chans=in_chans,
                                            channel_groups=grouped_bands,
                                            num_classes=-1,
                                            global_pool=False
                                            )
    elif 'dofa' in pretrain_tag:
        encoder = dofa.__dict__[name](img_size=im_size, 
                                      patch_size=patch_size, 
                                      num_classes=-1,
                                      global_pool=False
                                      )
        # remove other head layers
        encoder.norm = torch.nn.Identity()
    else:
        # ordinary vit from timm
        # e.g., name = 'vit_base_patch16_384'
        encoder = timm.create_model(name, pretrained=False, 
                                    global_pool='', num_classes=-1, # Do not load classifier head 
                                    in_chans=in_chans, img_size=im_size, patch_size=patch_size)
        # remove other head layers
        encoder.norm = torch.nn.Identity()
        encoder.head_drop = torch.nn.Identity()
    
    return encoder