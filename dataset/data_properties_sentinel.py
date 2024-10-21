import numpy as np

from fastai.vision.all import *
from albumentations.pytorch import ToTensorV2
import albumentations as A


IMG_CLIP = True
IMG_CLIP_SCALE = 3

S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

S2A_MEAN = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945, 2517.76053101, 2581.64687018, 2645.51888987, 2368.51236873, 1805.06846033]
S2A_STD = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101, 1474.78900051, 1439.3086061, 1582.28010962, 1455.52084939, 1343.48379601]

S2C_MEAN = [1605.57504906, 1390.78157673, 1314.8729939, 1363.52445545, 1549.44374991, 2091.74883118, 2371.7172463, 2299.90463006, 2560.29504086, 830.06605044, 22.10351321, 2177.07172323, 1524.06546312]
S2C_STD = [786.78685367, 850.34818441, 875.06484736, 1138.84957046, 1122.17775652, 1161.59187054, 1274.39184232, 1248.42891965, 1345.52684884, 577.31607053, 51.15431158, 1336.09932639, 1136.53823676]

BANDS ={3:  [3,2,1],
        4:  [3,2,1,8],  # 8: nir band
        9:  [3,2,1,4,5,6,7,11,12], # satlas/dofa bands
        10: [3,2,1,4,5,6,7,8,11,12],
        12: [3,2,1,0,4,5,6,7,8,9,11,12], # remove B10 - cirrus band
        13: [3,2,1,0,4,5,6,7,8,9,10,11,12]}
BANDS_2A ={3:  [3,2,1],
           4:  [3,2,1,8],  # 8: nir band
           9:  [3,2,1,4,5,6,7,10,11], # satlas/dofa bands
           10: [3,2,1,4,5,6,7,8,10,11],
           12: [3,2,1,0,4,5,6,7,8,9,10,11]}

WAVELENS_S2 = np.array([0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.375, 1.61, 2.19])
WAVELENS_S2A = np.array([0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.61, 2.19])
WAVELENS_S1 = np.array([3.75, 3.75])

SDTYPE = {1:np.float32, 2:np.uint16}

# CLS_DICT = {0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub and Scrub', 6:'Built area', 7:'Bare land', 8:'Ice and Snow'}
            
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


def data_normalize_s2(img, mode='s2_l1c'):
    means = S2C_MEAN if mode=='s2_l1c' else S2A_MEAN
    stds = S2C_STD if mode=='s2_l1c' else S2A_STD
    bands = BANDS if mode=='s2_l1c' else BANDS_2A
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

# band-wise data clip
def img_clip(img, mean, std, scale=2):
    min_value = mean - scale * std
    max_value = mean + scale * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# s2 - MS image data clip
def data_clip_s2(img):
    img = img.astype(np.float32)
    
    # get band info
    n_bands = img.shape[0]
    band_ids = BANDS[n_bands]
    
    # data clip
    img_new = []
    for b in range(n_bands):
        bi = band_ids[b]
        img0 = img[b]
        img0 = img_clip(img0, S2C_MEAN[bi], S2C_STD[bi])
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