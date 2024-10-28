import pickle
import numpy as np
from torch.utils.data import Dataset
import lmdb

from dataset.data_properties_sentinel import *


class GHSLDataset(Dataset):
    def __init__(self, lmdb_file, mode='s2_l1c', n_bands=13, N=-1,
                 transform=None, s2_transform=None, s2_normalize=True):
        assert mode=='s2_l1c', f"Mode {mode} is not supported for GHSL dataset! (only 's2_l1c' is supported)"
        if n_bands not in list(BANDS.keys()):
            n_bands = 13
        self.n_bands = n_bands
        self.mode = mode
        self.lmdb_file = lmdb_file
        # self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        # self.s1_normalize = s1_normalize
        self.s2_normalize = s2_normalize
        # self.n_prompts = n_prompts if type(n_prompts)==type([]) else [n_prompts]
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            total_N = int(txn.stat()['entries']//2)
            if N<0:
                self.length = total_N
            else: 
                assert N<=total_N, f"Requested number of samples {N} is larger than the total number of samples {total_N} in the dataset!"
                self.length = N        
                print(f'Use a subset of {N} samples from the dataset.')            
        
        self.transform = SegAugTransforms(transform)

    def __getitem__(self, index):
        # get data
        with self.env.begin(write=False) as txn:
            # read labels
            key_lbl = f'{index}_bc'
            data_lbl = txn.get(key_lbl.encode())
            lbl_bytes, lbl_shape, lbl_fid = pickle.loads(data_lbl)
            lbl_ = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
            lbl_.setflags(write=True)
            # reset class index
            lbl = np.ones_like(lbl_)*MASK_CODE
            # set 1-4 to 0, 11-15 to 1, 21-25 to 2
            source_codes = [1,2,3,4,11,12,13,14,15,21,22,23,24,25]
            target_codes = [0,0,0,0,1,1,1,1,1,2,2,2,2,2]
            for i, ind in zip(target_codes, source_codes):
                lbl[lbl_==ind] = i    
            # read images
            key_s = f'{index}_{self.mode}'
            data_s = txn.get(key_s.encode())
            img_s, shape_s, fid_s = pickle.loads(data_s)
            assert lbl_fid==fid_s, f"The fid of the image ({fid_s}) and its label ({lbl_fid}) are inconsist!"
            assert shape_s[-1]==lbl_shape[-1], f"The shape of the image and its label of pair {index} are inconsist!"
            img = np.frombuffer(img_s, dtype=SDTYPE[2]).reshape(shape_s)
        
        # data processing for s2c
        # get specific bands for s2
        img = img[BANDS[self.n_bands]]

        if self.s2_normalize:
            # normalize with mean and std
            img = data_normalize_s2(img).astype(np.float32)
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
                KEY_DATASAT: 'ghsl',
                KEY_WAVE: waves,
                KEY_WEAKLBL: True}
    
    def __len__(self):
        return self.length

