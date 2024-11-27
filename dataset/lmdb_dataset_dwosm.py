# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:04 2023

@author: liu_ch
"""

import os
import random
import pickle
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import lmdb, torch
from dataset.data_properties_sentinel import *


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


def s1_exclude_outliers(img, head_rate=1, tail_rate=99):
    head = np.percentile(img, head_rate)
    tail = np.percentile(img, tail_rate, axis=(0, 1))
    img = np.clip(img, head, tail)
    return img
                

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
        # self.transform = transform
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
        
        self.transform = SegAugTransforms(transform)


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
                KEY_WAVE: waves,
                KEY_WEAKLBL: False}
    
    def __len__(self):
        return self.length
    
    
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



