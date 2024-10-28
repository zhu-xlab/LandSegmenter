import pickle
import numpy as np
from torch.utils.data import Dataset
import lmdb

from dataset.data_properties_landsat import *


LBL_KEYS = ['lc', 'imp']

class NLCDDataset(Dataset):
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
                print(f'Use a subset of {N} samples from the dataset.')            
        
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
                    # original: [11 12 21 22 23 24 31 41 42 43 52 71 81 82 90 95]
                    # new:      [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
                    origin_inds = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
                    for i, ind in enumerate(origin_inds):
                        lbl = np.where(lbl==ind, i, lbl)
                if lbl_mode == 'imp':
                    # original: [ 0   20 21 22 23 24 25 26 27 28 29]
                    # new:      [ 255  0  0  0  0  1  1  1  2  2  2]  255-Unclassified 0-road 1-non-road and non-energy impervious 2-energy impervious/product
                    for i, ind in zip([255, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]):
                        lbl = np.where(lbl==ind, i, lbl)
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
        img, lbl1, lbl2 = self.transform((img, lbls[0], lbls[1]))                
        lbl = torch.stack([lbl1, lbl2], dim=0)

        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'nlcd',
                KEY_WAVE: waves,
                KEY_WEAKLBL: True}
    
    def __len__(self):
        return self.length

