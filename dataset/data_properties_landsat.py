import numpy as np

from fastai.vision.all import *
from albumentations.pytorch import ToTensorV2
import albumentations as A

IMG_KEYS = ['toa', 'sr']
SDTYPE = {'toa':np.float32, 'sr':np.uint16}
IMG_CLIP = True
IMG_CLIP_SCALE = 3

TOA_MEAN = [0.139721155, 0.125081092, 0.123868674, 0.131810322, 0.283200920, 0.246309161,
            0.171668261, 0.124766774, 0.00241818046, 302.881165, 301.475922]
TOA_STD  = [0.048360974, 0.054451246, 0.067494757, 0.099091828, 0.098961689, 0.13054624, 
            0.12400048,  0.077818833, 0.0047299736, 10.850907, 10.487716]
SR_MEAN  = [9129.262,  9584.849, 10972.924, 11666.224, 17488.928, 16536.77,  13952.874] 
SR_STD   = [2375.5134, 2545.028, 2975.2673, 4044.3245,  3713.504, 4925.688,  4821.258 ]

BANDS_TOA = {3:  [3,2,1],
             7:  [3,2,1,0,4,5,6],
             11: [3,2,1,0,4,5,6,7,8,9,10]}
BANDS_SR = {3: [3,2,1],
            7: [3,2,1,0,4,5,6]}

WAVELENS = np.array([0.443, 0.482, 0.561, 0.655, 0.865, 1.610, 2.200, 0.590, 1.373, 10.895, 12.005])

KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
KEY_WEAKLBL = 'weak'
MASK_CODE = 255


def single_band_normalize(img, mean, std, clip=False):
    if clip:
        scale = IMG_CLIP_SCALE
        min_value = mean - scale * std
        max_value = mean + scale * std
        img = np.clip(img, min_value, max_value)
    img = (img-mean)/std
    return img

def data_normalize(img, mode='toa'):
    means = TOA_MEAN if mode=='toa' else SR_MEAN
    stds = TOA_STD if mode=='toa' else SR_STD
    bands = BANDS_TOA if mode=='toa' else BANDS_SR
    # get band info
    n_bands = img.shape[0]
    band_ids = bands[n_bands]
    
    # data normalize
    img_new = []
    for b in range(n_bands):
        img0 = single_band_normalize(img[b], means[band_ids[b]], stds[band_ids[b]], clip=IMG_CLIP)
        img_new.append(img0[None])
    
    # concatenate
    img_new = np.concatenate(img_new, axis=0)
    return img_new


# augmentation class for (img, label) pairs
class SegAugTransforms(ItemTransform):
    def __init__(self, aug): 
        if aug is None:
            self.aug = ToTensorV2()
        else:
            self.aug = A.Compose(aug.transforms + [ToTensorV2()])
    def encodes(self, x):
        img, lbl = x
        if lbl.flags['WRITEABLE'] is False:
            lbl_ = np.copy(lbl)
            lbl_.setflags(write=True)
        else:
            lbl_ = lbl
        # for albumentations to work correctly, the channels must be at the last dimension
        tens = self.aug(image=img.transpose(1,2,0), mask=lbl)
        return tens['image'], tens['mask']

# augmentation class for (img, label, label) triplets
class SegAugTransformsTwoMasks(ItemTransform):
    def __init__(self, aug): 
        if aug is None:
            self.aug = A.Compose([ToTensorV2()], 
                                 additional_targets={'mask1': 'mask', 'mask2': 'mask'})
        else:
            self.aug = A.Compose(aug.transforms + [ToTensorV2()],  # Use the list of transforms from trans_tr
                                additional_targets={'mask1': 'mask', 'mask2': 'mask'}
                                )
    def encodes(self, x):
        img, lbl1, lbl2 = x
        if lbl1.flags['WRITEABLE'] is False:
            lbl1_ = np.copy(lbl1)
            lbl1_.setflags(write=True)
        else:
            lbl1_ = lbl1
        if lbl2.flags['WRITEABLE'] is False:
            lbl2_ = np.copy(lbl2)
            lbl2_.setflags(write=True)
        # for albumentations to work correctly, the channels must be at the last dimension
        data_aug = self.aug(image=img.transpose(1,2,0), mask1=lbl1, mask2=lbl2)
        return data_aug['image'], data_aug['mask1'], data_aug['mask2']