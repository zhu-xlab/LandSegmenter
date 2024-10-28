import pickle
import numpy as np
from torch.utils.data import Dataset
import lmdb

from dataset.data_properties_sentinel import *

BANDS_2A10 = {10: [2,1,0,3,4,5,6,7,8,9],  # 10 bands
              3:  [2,1,0],  # 3 bands
              }

class MultiSenGeDataset(Dataset):
    def __init__(self, lmdb_file, n_bands=10, N=-1,
                 transform=None, s2_transform=None, s2_normalize=True):
        if n_bands not in list(BANDS_2A10.keys()):
            n_bands = 10
        self.n_bands = n_bands
        self.lmdb_file = lmdb_file
        self.s2_transform = s2_transform
        self.s2_normalize = s2_normalize
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            total_N = txn.stat()['entries']
            if N<0:
                self.length = total_N
            else: 
                assert N<=total_N, f"Requested number of samples {N} is larger than the total number of samples {total_N} in the dataset!"
                self.length = N        
                # print(f'Use a subset of {N} samples from the dataset.')            
        
        self.transform = SegAugTransforms(transform)

    def __getitem__(self, index):
        # get data
        with self.env.begin(write=False) as txn:
            # read data
            data = txn.get(str(index).encode())
            img_bytes, img_shape, lbl_bytes, lbl_shape, fid = pickle.loads(data)
            # read labels
            lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).squeeze().copy()
            lbl.setflags(write=True)
            lbl = lbl - 1
            # read images
            assert img_shape[-1]==lbl_shape[-1], f"The shape of the image and its label of pair {index}-{fid.decode()} are inconsist!"
            img = np.frombuffer(img_bytes, dtype=SDTYPE[2]).reshape(img_shape)
        
        # data processing for s2c
        # get specific bands for s2
        img = img[BANDS_2A10[self.n_bands]]

        if self.s2_normalize:
            # normalize with mean and std
            img = data_normalize_s2(img, mode='s2_l2a').astype(np.float32)
        else:
            # clip data (mean-2std,mean+2std) and normalize to [0,255]
            img = data_clip_s2(img)
            # /255 to [0,1]
            img = img.astype(np.float32)/255
        
        # transforms for separate modes
        if self.s2_transform is not None:
            img = self.s2_transform(img)
        
        # wavelengths
        waves = WAVELENS_S2[BANDS[self.n_bands]]

        # transform for (img,lbl) pairs
        img, lbl = self.transform((img, lbl))                
        
        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'multisenge',
                KEY_WAVE: waves,
                KEY_WEAKLBL: False}
    
    def __len__(self):
        return self.length