# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:04 2023

@author: liu_ch
"""

import os
import random
import pickle
import numpy as np
import albumentations as A
from fastai.vision.all import *
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import lmdb, torch

# import matplotlib as mpl
# import matplotlib.pyplot as plt

### band statistics: mean & std
# calculated from 50k subset
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

WAVELENS_S2 = np.array([0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.375, 1.61, 2.19])
WAVELENS_S1 = np.array([3.75, 3.75])

SDTYPE = {1:np.float32, 2:np.uint16}

CLS_DICT = {0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub and Scrub', 6:'Built area', 7:'Bare land', 8:'Ice and Snow'}
            
KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
MASK_CODE = 255

# band-wise data clip
def img_clip(img, mean, std, scale=2):
    min_value = mean - scale * std
    max_value = mean + scale * std
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# s1 - SAR image data clip
def data_clip_s1(img):
    # get band info
    n_bands = img.shape[0]
    
    # data clip
    img_new = []
    for b in range(n_bands):
        img0 = img[b]
        img0 = img_clip(img0, S1_MEAN[b], S1_STD[b], scale=2)
        img_new.append(img0[None])
    
    # concatenate
    img_new = np.concatenate(img_new, axis=0)
    return img_new


def data_normalize_s1(img):
    # get band info
    n_bands = img.shape[0]
    
    # data clip
    img_new = []
    for b in range(n_bands):
        # exclude outliers
        img0 = s1_exclude_outliers(img[b], head_rate=1, tail_rate=99)
        img0 = single_band_normalize(img0, S1_MEAN[b], S1_STD[b])
        img_new.append(img0[None])
    
    # concatenate
    img_new = np.concatenate(img_new, axis=0)
    return img_new


def data_normalize_s2(img):
    # get band info
    n_bands = img.shape[0]
    band_ids = BANDS[n_bands]
    
    # data normalize
    img_new = []
    for b in range(n_bands):
        img0 = single_band_normalize(img[b], S2C_MEAN[band_ids[b]], S2C_STD[band_ids[b]], clip=False)
        img_new.append(img0[None])
    
    # concatenate
    img_new = np.concatenate(img_new, axis=0)
    return img_new


def s1_exclude_outliers(img, head_rate=1, tail_rate=99):
    head = np.percentile(img, head_rate)
    tail = np.percentile(img, tail_rate, axis=(0, 1))
    img = np.clip(img, head, tail)
    return img
    
    
def single_band_normalize(img, mean, std, clip=False):
    if clip:
        scale = 3
        min_value = mean - scale * std
        max_value = mean + scale * std
        img = np.clip(img, min_value, max_value)
    img = (img-mean)/std
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
    

# /10000 normalization
def normalize_s2(img):
    img = img.astype(np.float32)
    
    # get band info
    n_bands = img.shape[0]
    # band_ids = BANDS[n_bands]
    
    # normalize band by band
    img_new = []
    for b in range(n_bands):
        im = img[b]
        
        # # normalize
        im /= 10000.0
        im = np.clip(im, 0, 1)
        
        # concatenate
        img_new.append(im[None])
    
    # concatenate
    img_new = np.concatenate(img_new, axis=0)
    # print(img_new.shape, img_new.dtype, img_new.min(), img_new.max())
    
    return img_new
                

# get subset of given dataset    
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


# construct subset with randomly generated indexes
def random_subset(dataset, f_or_num, seed=None):
    if f_or_num <=0:
        raise ValueError('Please provide valid subset size!')
    elif f_or_num <= 1:
        n_select = int(len(dataset)*f_or_num)
    else:
        n_select = int(f_or_num)
    indices_all = np.arange(len(dataset))
    if seed is not None: random.seed(seed)
    random.shuffle(indices_all)
    indices = indices_all[:n_select]
    return Subset(dataset, indices)


# split all the data into training and validation sets
def split_for_train_and_val_sets(dataset, nval, seed=None):
    # get validation set number
    if nval <=0:
        raise ValueError('Please provide valid subset size!')
    elif nval <= 1:
        n_val = int(len(dataset)*nval)
    else:
        n_val = int(nval)
    # split dataset
    if n_val>0:
        n_tr = len(dataset)-n_val
        train_set, val_set = random_split(dataset, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    else:
        train_set, val_set = dataset, None
    return train_set, val_set


class DWOSMDataset(Dataset):
    def __init__(self, lmdb_file, lbl_type='dw', mode='s2', n_bands=13, 
                 transform=None, s1_transform=None, s2_transform=None, 
                 s1_normalize=True, s2_normalize=True):
        if n_bands not in list(BANDS.keys()):
            n_bands = 13
        self.n_bands = n_bands
        assert lbl_type in ['dw','osm'], f'Please set lbl_type to \'dw\' or \'osm\'! Current: {lbl_type}'
        self.lbl_type = lbl_type
        self.mode = mode
        self.lmdb_file = lmdb_file
        self.transform = transform
        self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        self.s1_normalize = s1_normalize
        self.s2_normalize = s2_normalize
        # self.n_prompts = n_prompts if type(n_prompts)==type([]) else [n_prompts]
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            bsname = os.path.basename(os.path.normpath(self.lmdb_file))
            if 'osm' in bsname:
                self.length = int(txn.stat()['entries']/4)
            else: 
                self.length = int(txn.stat()['entries']/3)                    
        
        if self.transform is None:
            t = ToTensorV2()
            def base_transform(x):
                img, lbl = x
                if lbl.flags['WRITEABLE'] is False:
                    lbl_writable = np.copy(lbl)
                    lbl_writable.setflags(write=True)
                    tens = t(image=img.transpose(1,2,0), mask=lbl_writable)
                else:
                    tens = t(image=img.transpose(1,2,0), mask=lbl)
                return tens['image'], tens['mask']
            self.transform = base_transform
        

    def __getitem__(self, index):
        # get data
        with self.env.begin(write=False) as txn:
            # read labels
            if self.lbl_type == 'dw':
                key_lbl = f'dw_{index}'
            else:
                key_lbl = f'osm_{index}'
            data_lbl = txn.get(key_lbl.encode())
            lbl_bytes, lbl_shape, lbl_fid = pickle.loads(data_lbl)
            lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
            lbl.setflags(write=True)
            # remove cloud class
            if self.lbl_type == 'dw':
                lbl[lbl==10] = 0
            # convert background to 255
            lbl[lbl==0] = 255
            lbl[lbl<255] = lbl[lbl<255] - 1
            # read images
            imgs = {}
            for s in [1, 2]:
                if str(s) in self.mode: 
                    key_s = f's{s}_{index}'
                    data_s = txn.get(key_s.encode())
                    img_s, shape_s, fid_s = pickle.loads(data_s)
                    assert shape_s[-1]==lbl_shape[-1], f"The shape of the s{s} image and its label of pair {index} are inconsist!"
                    assert fid_s[3:]==lbl_fid[len(self.lbl_type)+1:], f"The fid of the s{s} image and its label of pair are inconsist!"
                    img = np.frombuffer(img_s, dtype=SDTYPE[s]).reshape(shape_s)
                    imgs[s] = img
        
        # data processing for s1
        if '1' in self.mode:
            img = imgs[1]
            if self.s1_normalize:
                # normalize with mean and std
                img = data_normalize_s1(img)
            else:
                # clip data (mean-2std,mean+2std) and normalize to [0,255]
                img = data_clip_s1(img)
                imgs[1] = img.astype(np.float32)/255
        
        # data processing for s2
        if '2' in self.mode:
            img = imgs[2]
            # get specific bands for s2
            img = img[BANDS[self.n_bands]]

            if self.s2_normalize:
                # normalize with mean and std
                imgs[2] = data_normalize_s2(img).astype(np.float32)
            else:
                # clip data (mean-2std,mean+2std) and normalize to [0,255]
                img = data_clip_s2(img)
                # /255 to [0,1]
                imgs[2] = img.astype(np.float32)/255
        
        # transforms for separate modes
        if '1' in self.mode and self.s1_transform is not None:
            imgs[1] = self.s1_transform(imgs[1])
        if '2' in self.mode and self.s2_transform is not None:
            imgs[2] = self.s2_transform(imgs[2])
        
        # concatenate s1 and s2
        if self.mode == 's1':
            img = imgs[1]
            waves = WAVELENS_S1
        elif self.mode == 's2': 
            img = imgs[2]
            waves = WAVELENS_S2[BANDS[self.n_bands]]
        else:
            img = np.concatenate([imgs[1], imgs[0]], axis=0)  # s2 then s1
            waves = np.hstack([WAVELENS_S2[BANDS[self.n_bands]],WAVELENS_S1])
        
        # transform for (img,lbl) pairs
        img, lbl = self.transform((img, lbl))                
        
        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: self.lbl_type,
                KEY_WAVE: waves}
    
    def __len__(self):
        return self.length
    

# augmentation class for (img,label) pairs
class SegAugTransforms(ItemTransform):
    def __init__(self, aug): 
        self.aug = aug
        self.ToTensor = ToTensorV2()
    def encodes(self, x):
        img, lbl = x
        # for albumentations to work correctly, the channels must be at the last dimension
        data_aug = self.aug(image=img.transpose(1,2,0), mask=lbl)
        if data_aug['mask'].flags['WRITEABLE'] is False:
            lbl_writable = np.copy(data_aug['mask'])
            lbl_writable.setflags(write=True)
        else: 
            lbl_writable = data_aug['mask']
        tens = self.ToTensor(image=data_aug['image'], mask=lbl_writable)
        return tens['image'], tens['mask']
    
    
if __name__ == '__main__':
    selected_i = 0
    si = 2
    lt = 'dw'
    dw_train = '/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_dw_osm_lmdb/dw_train_lmdb'
    dw_test = '/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_dw_osm_lmdb/dw_test_lmdb'
    # dw_train = '/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_dw_osm_lmdb/osm_train_small_lmdb'
    # dw_test = '/p/project/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_dw_osm_lmdb/osm_test_lmdb'
    val_dataset = DWOSMDataset(
        lmdb_file = dw_train,
        mode = 's2',
        lbl_type = 'dw',
        n_bands = 13,
        s2_normalize = True
        )
    test_dataset = DWOSMDataset(
        lmdb_file = dw_test,
        mode = 's12',
        lbl_type = lt,
        n_bands = 13,
        s2_normalize = True
        )
    
    print(len(val_dataset), len(test_dataset))
    
    bs = 10
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=0)
    
    for i, s2c in enumerate(val_loader):
        if i == selected_i:
            img, lbl, da, wave = s2c[KEY_IMG], s2c[KEY_LBL], s2c[KEY_DATASAT], s2c[KEY_WAVE]
            print(img.shape, img.dtype, img.mean(), img.min(), img.max(), lbl.shape, lbl.dtype, np.unique(lbl))
            print(da)
            print(wave.shape, wave.dtype)
            print(wave[0])
            # plt.figure(figsize=(10,5))
            # plt.subplot(121)
            # # plt.imshow((img.numpy()[si,[3,2,1]]).transpose(1,2,0))
            # plt.imshow((img[si].numpy()).transpose(1,2,0))
            # plt.subplot(122)
            # plt.imshow(lbl[si].numpy(),norm=mpl.colors.Normalize(vmin=0,vmax=8))
            break
    
    # for i, s2c in enumerate(test_loader):
    #     if i == selected_i:
    #         img, lbl = s2c['img'], s2c['gt']
    #         print(img.shape, img.dtype, img.mean(), img.min(), img.max(), lbl.shape, lbl.dtype, np.unique(lbl))
    #         print(img[:2].mean(), img[:2].std())
    #         # plt.figure(figsize=(10,5))
    #         # plt.subplot(121)
    #         # # plt.imshow((img.numpy()[si,[3,2,1]]).transpose(1,2,0))
    #         # plt.imshow((img[si][[5,4,3]].numpy()).transpose(1,2,0))
    #         # plt.subplot(122)
    #         # plt.imshow(lbl[si].numpy(),norm=mpl.colors.Normalize(vmin=0,vmax=8))
    #         # plt.figure(figsize=(10,5))
    #         # plt.subplot(121)
    #         # plt.imshow((img[si][0].numpy()))
    #         # plt.subplot(122)
    #         # plt.imshow((img[si][1].numpy()))
    #         break



