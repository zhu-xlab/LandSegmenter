import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset

# from albumentations.pytorch import ToTensorV2
from dataset.data_properties_sentinel import SegAugTransforms

 
MEANS = [123.675, 116.28, 103.53] # ImageNet stats 
STDS = [58.395, 57.12, 57.375] # ImageNet stats 
WAVES = np.array([0.665, 0.56, 0.49])  # RGB

KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
KEY_WEAKLBL = 'weak'
MASK_CODE = 255

def band_normalize(band, bi):
    mean_v = MEANS[bi]
    std_v = STDS[bi]
    return (band - mean_v) / std_v
    
def band_clip(band, bi):
    mean_v = MEANS[bi]
    std_v = STDS[bi]
    min_v, max_v = mean_v - 2*std_v, mean_v + 2*std_v
    b_ = np.clip(band, min_v, max_v)
    b_ = (b_ - min_v) / (max_v - min_v)
    return b_

class OpenEarthMapDataset(Dataset):
    def __init__(self, lmdb_file, normalize=True, transform=None):
        self.lmdb_file = lmdb_file
        # self.transform = transform
        self.normalize = normalize
        self.n_bands = 3
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()['entries'])
            
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
            # data key
            key = str(index)
            data = txn.get(key.encode())
            data_bytes, data_shape, lbl_bytes, lbl_shape, a = pickle.loads(data)
            img = np.frombuffer(data_bytes, dtype=np.uint8).reshape(data_shape).astype(np.float32) 
            lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
            lbl.setflags(write=True)
            
        # image processing
        if self.normalize:
            for i in range(self.n_bands):
                img[i] = band_normalize(img[i], i)
        else:
            for i in range(self.n_bands):
                img[i] = band_clip(img[i], i)
        
        # wave lengths
        waves = WAVES

        # set 0 to MASK_CODE
        lbl[lbl == 0] = MASK_CODE
        lbl[lbl != MASK_CODE] -= 1
        
        # transform for (img,lbl) pairs
        try:
            img, lbl = self.transform((img, lbl))                
        except:
            raise ValueError(f'Error in transforming data with key={key}, {a}.')

        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'openearthmap',
                KEY_WAVE: waves,
                KEY_WEAKLBL: False}
    
    def __len__(self):
        return self.length