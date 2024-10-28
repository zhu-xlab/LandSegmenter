import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset

from albumentations.pytorch import ToTensorV2
from dataset.data_properties_sentinel import SegAugTransforms


BANDS = {3: [2,1,0], 
         4: [2,1,0,3]}  # [2,1,3,0] corresponds to [red, green, blue, nir]
WAVES = np.array([0.480, 0.560, 0.655, 0.865])  # order:red, green, blue, nir
MEANS = [1042.59240722656, 915.618408203125, 671.260559082031, 2605.20922851562]  # order:red, green, blue, nir
STDS  = [957.958435058593, 715.548767089843, 596.943908691406, 1059.90319824218]

KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
KEY_WEAKLBL = 'weak'
MASK_CODE = 255
clip_scale = 3

def band_normalize(band, bi, clip=False):
    mean_v = MEANS[bi]
    std_v = STDS[bi]
    if clip:
        band = np.clip(band, mean_v - clip_scale*std_v, mean_v + clip_scale*std_v)
    return (band - mean_v) / std_v
    
def band_clip(band, bi):
    mean_v = MEANS[bi]
    std_v = STDS[bi]
    min_v, max_v = mean_v - 2*std_v, mean_v + 2*std_v
    b_ = np.clip(band, min_v, max_v)
    b_ = (b_ - min_v) / (max_v - min_v)
    return b_

class DyEarthDataset(Dataset):
    def __init__(self, lmdb_file, n_bands=4, normalize=True, transform=None, timestamp=None):
        if n_bands not in [3,4]:
            print(f'Invalid n_bands={n_bands}, set to 4.')
            n_bands = 4
        self.n_bands = n_bands
        self.lmdb_file = lmdb_file
        # self.transform = transform
        self.normalize = normalize
        self.timestamp = timestamp
        self.all_timestamps = [0,1,2,3,4,5]
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # bsname = os.path.basename(os.path.normpath(self.lmdb_file))
            self.length = int(txn.stat()['entries'])//6
        
        self.transform = SegAugTransforms(transform)
        # if self.transform is None:
        #     t = ToTensorV2()
        #     def base_transform(x):
        #         img, lbl = x
        #         if lbl.flags['WRITEABLE'] is False:
        #             lbl_writable = np.copy(lbl)
        #             lbl_writable.setflags(write=True)
        #             tens = t(image=img.transpose(1,2,0), mask=lbl_writable)
        #         else:
        #             tens = t(image=img.transpose(1,2,0), mask=lbl)
        #         return tens['image'], tens['mask']
        #     self.transform = base_transform

    def __getitem__(self, index):
        # get data
        with self.env.begin(write=False) as txn:
            # timestamp
            if self.timestamp is not None:
                if self.timestamp not in self.all_timestamps:
                    timestamp = np.random.choice(self.all_timestamps)
                else:
                    timestamp = self.timestamp
            else:
                timestamp = np.random.choice(self.all_timestamps)
            # data key
            key = f'{index}_{timestamp}'
            data = txn.get(key.encode())
            try:
                data_bytes, data_shape, lbl_bytes, lbl_shape, _ = pickle.loads(data)
            except:
                raise ValueError(f'Error in loading data with key={key}.')
            img = np.frombuffer(data_bytes, dtype=np.int16).reshape(data_shape).astype(np.float32)
            lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
            lbl.setflags(write=True)
            
        # image processing
        img = img[BANDS[self.n_bands]]
        if self.normalize:
            for i in range(self.n_bands):
                img[i] = band_normalize(img[i], i, clip=False)
        else:
            for i in range(self.n_bands):
                img[i] = band_clip(img[i], i)
        
        # wave lengths
        waves = WAVES[BANDS[self.n_bands]] 
        
        # transform for (img,lbl) pairs
        img, lbl = self.transform((img, lbl))                
        
        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'dynamicearthnet',
                KEY_WAVE: waves,
                KEY_WEAKLBL: False}
    
    def __len__(self):
        return self.length