import pickle
import numpy as np
from torch.utils.data import Dataset
import lmdb

from dataset.data_properties_landsat import *

# LBL_KEYS = ['lc', 'lu']
LBL_KEYS = ['lu']

class USFSDataset(Dataset):
    def __init__(self, lmdb_file=None,
                 img_mode='toa', n_bands='all', N=-1,
                 transform=None, img_transform=None, img_normalize=True):
        assert img_mode in IMG_KEYS+['rand'], f"Mode {img_mode} is not supported!"
        if img_mode == 'rand':
            assert n_bands in BANDS_SR.keys(), f"Number of bands {n_bands} is not supported for random mode (must be less than 7, the number of SR image)!"
        elif img_mode == 'toa':
            if n_bands == 'all': n_bands = 11
            if n_bands not in list(BANDS_TOA.keys()): n_bands = 11
        elif img_mode == 'sr':
            if n_bands == 'all': n_bands = 7
            if n_bands not in list(BANDS_SR.keys()): n_bands = 7
        self.n_bands = n_bands
        self.img_mode = img_mode
        self.lmdb_file = lmdb_file
        self.img_transform = img_transform
        self.img_normalize = img_normalize
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            total_N = int(txn.stat()['entries']//4)
            if N<0:
                self.length = total_N
            else: 
                assert N<=total_N, f"Requested number of samples {N} is larger than the total number of samples {total_N} in the dataset!"
                self.length = N        
                # print(f'Use a subset of {N} samples from the dataset.')          
                  
        if len(LBL_KEYS) == 1:
            self.transform = SegAugTransforms(transform)
        else:
            self.transform = SegAugTransformsTwoMasks(transform)
        

    def __getitem__(self, index):
        # get image data type
        if self.img_mode == 'rand':
            mode = np.random.choice(IMG_KEYS)
        else:
            mode = self.img_mode
        # get data
        with self.env.begin(write=False) as txn:
            # read labels
            lbls = []
            for lbl_mode in LBL_KEYS:
                key_lbl = f'{index}_{lbl_mode}'
                data_lbl = txn.get(key_lbl.encode())
                lbl_bytes, lbl_shape, lbl_fid = pickle.loads(data_lbl)
                lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
                lbl.setflags(write=True)
                # reset label index
                if lbl_mode == 'lc':
                    # original: [ 1  3  4  5  7  8  9 10 11 12 13 14 15]  15: non-processing area mask
                    # new:      [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
                    lbl = np.where(lbl>1, lbl-2, lbl-1)
                    lbl = np.where(lbl>4, lbl-1, lbl)
                    lbl[lbl>11] = MASK_CODE
                if lbl_mode == 'lu':
                    # original: [ 1  2  3  4  5  6  7]
                    # new:      [ 0  1  2  3 255 4 255]
                    lbl[lbl==5] = 7
                    lbl = np.where(lbl>5, lbl-2, lbl-1) 
                    lbl[lbl>4] = MASK_CODE
                lbls.append(lbl)
                    
            # read images
            key_s = f'{index}_{mode}'
            data_s = txn.get(key_s.encode())
            img_s, shape_s, fid_s = pickle.loads(data_s)
            assert lbl_fid==fid_s, f"The fid of the image ({fid_s}) and its label ({lbl_fid}) are inconsist!"
            assert shape_s[-1]==lbl_shape[-1], f"The shape of the image and its label of pair {index} are inconsist!"
            img = np.frombuffer(img_s, dtype=SDTYPE[mode]).reshape(shape_s)

        # data processing for img
        # get specific bands for s2
        bands = BANDS_TOA if mode=='toa' else BANDS_SR
        img = img[bands[self.n_bands]]

        if self.img_normalize:
            # normalize with mean and std
            img = data_normalize(img, mode=mode).astype(np.float32)
        else:
            raise NotImplementedError('Simple data normalization is not implemented yet!')
        
        # transforms for separate modes
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        # wavelengths
        waves = WAVELENS[bands[self.n_bands]]

        # transform for (img,lbl) pairs
        if len(lbls) == 1:
            img, lbl = self.transform((img, lbls[0]))
        else:
            img, lbl1, lbl2 = self.transform((img, lbls[0], lbls[1]))               
            lbl = torch.stack([lbl1, lbl2], axis=0)

        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'usfs',
                KEY_WAVE: waves,
                KEY_WEAKLBL: True}
    
    def __len__(self):
        return self.length

