import pickle
import numpy as np
from fastai.vision.all import *
from torch.utils.data import Dataset
import lmdb

from dataset.data_properties_sentinel import *

IMG_KEYS = ['s2_l1c', 's2_l2a']


class WorldCoverDataset(Dataset):
    def __init__(self, lmdb_file_v100=None, lmdb_file_v200=None, 
                 mode='rand', n_bands='all', N=-1,
                 transform=None, s2_transform=None, s2_normalize=True):
        assert mode in IMG_KEYS+['rand'], f"Mode {mode} is not supported!"
        if n_bands not in list(BANDS.keys())+['all']:
            n_bands = 'all'
        self.n_bands = n_bands
        self.mode = mode
        self.versions = []
        self.lmdb_files = {}
        if lmdb_file_v100 is not None:
            self.versions.append('v100')
            self.lmdb_files['v100'] = lmdb_file_v100
        if lmdb_file_v200 is not None:
            self.versions.append('v200')
            self.lmdb_files['v200'] = lmdb_file_v200
        # self.s1_transform = s1_transform
        self.s2_transform = s2_transform
        # self.s1_normalize = s1_normalize
        self.s2_normalize = s2_normalize
        # self.n_prompts = n_prompts if type(n_prompts)==type([]) else [n_prompts]
        
        # open lmdb file
        self.envs = {}
        for v in self.versions:
            self.envs[v] = lmdb.open(self.lmdb_files[v], max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.envs[v].begin(write=False) as txn:
            total_N = int(txn.stat()['entries']//3)
            if N<0:
                self.length = total_N
            else: 
                assert N<=total_N, f"Requested number of samples {N} is larger than the total number of samples {total_N} in the dataset!"
                self.length = N        
                # print(f'Use a subset of {N} samples from the dataset.')            
        
        self.transform = SegAugTransforms(transform)
        

    def __getitem__(self, index):
        # get version
        if len(self.versions)==1:
            v = self.versions[0]
        else:
            v = np.random.choice(self.versions)
        # get image data type
        if self.mode == 'rand':
            mode = np.random.choice(IMG_KEYS)
        else:
            mode = self.mode
        # get data
        with self.envs[v].begin(write=False) as txn:
            # read labels
            key_lbl = f'{index}_lc'
            data_lbl = txn.get(key_lbl.encode())
            lbl_bytes, lbl_shape, lbl_fid = pickle.loads(data_lbl)
            lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
            lbl.setflags(write=True)
            # convert label indexes from 0-10
            lbl[lbl>90] = lbl[lbl>90]+10
            lbl = np.floor(lbl/10).astype(np.uint8)
            lbl = lbl - 1
            # read images
            key_s = f'{index}_{mode}'
            data_s = txn.get(key_s.encode())
            img_s, shape_s, fid_s = pickle.loads(data_s)
            assert lbl_fid==fid_s, f"The fid of the image ({fid_s}) and its label ({lbl_fid}) are inconsist!"
            assert shape_s[-1]==lbl_shape[-1], f"The shape of the image and its label of pair {index} are inconsist!"
            img = np.frombuffer(img_s, dtype=SDTYPE[2]).reshape(shape_s)
        
        # data processing for s2c
        # get specific bands for s2
        if mode=='s2_l1c':
            n_bands = 13 if self.n_bands == 'all' else self.n_bands
            bands = BANDS[n_bands]
            waves = WAVELENS_S2[bands]
        else:
            assert mode=='s2_l2a', f"Mode {mode} is not supported!"
            n_bands = 12 if self.n_bands == 'all' else self.n_bands
            bands = BANDS_2A[n_bands]
            waves = WAVELENS_S2A[bands]
        img = img[bands]

        # normalize with mean and std
        if self.s2_normalize:
            # normalize with mean and std
            img = data_normalize_s2(img, mode=mode).astype(np.float32)
        else:
            # clip data (mean-2std,mean+2std) and normalize to [0,255]
            img = data_clip_s2(img)
            # /255 to [0,1]
            img = img.astype(np.float32)/255
        
        # transforms for separate modes
        if self.s2_transform is not None:
            img = self.s2_transform(img)

        # transform for (img,lbl) pairs
        img, lbl = self.transform((img, lbl))                
        
        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'worldcover',
                KEY_WAVE: waves,
                KEY_WEAKLBL: True}
    
    def __len__(self):
        return self.length


if __name__ == '__main__':
    fp = '/p/scratch/hai_ws4eo/liu_ch/data/WeakFT/weak_lmdbs/iran.lmdb'
    dataset = WorldCoverDataset(fp, mode='s2_l1c', n_bands='all', N=100)