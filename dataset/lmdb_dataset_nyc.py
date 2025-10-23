import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset

# from albumentations.pytorch import ToTensorV2
from dataset.data_properties_sentinel import SegAugTransforms

# NYC statistics
MEANS = [111.34724539, 130.45546838, 130.80935808, 156.86974368]
STDS = [46.64435224, 36.81250866, 32.99007486, 41.4493565 ]

WAVES = np.array([0.665, 0.56, 0.49, 0.83])  # RGB-NIR

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


class NYCDataset(Dataset):
    def __init__(self, lmdb_file, n_bands=4, normalize=True, transform=None):
        self.lmdb_file = lmdb_file
        self.normalize = normalize
        assert n_bands in [3, 4], 'NYC dataset has 3 or 4 bands.'
        self.n_bands = n_bands
        
        # open lmdb file
        self.env = lmdb.open(self.lmdb_file, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int(txn.stat()['entries'])
            
        self.transform = SegAugTransforms(transform)
        

    def __getitem__(self, index):
        # get data
        with self.env.begin(write=False) as txn:
            # data key
            key = str(index)
            data = txn.get(key.encode())
            data_bytes, data_shape, lbl_bytes, lbl_shape, a = pickle.loads(data)
            img = np.frombuffer(data_bytes, dtype=np.uint8).reshape(data_shape).astype(np.float32)[:self.n_bands]
            lbl = np.frombuffer(lbl_bytes, dtype=np.uint8).reshape(lbl_shape).copy()
            lbl.setflags(write=True)
            lbl = lbl-1
            
        # image processing
        if self.normalize:
            for i in range(self.n_bands):
                img[i] = band_normalize(img[i], i)
        else:
            for i in range(self.n_bands):
                img[i] = band_clip(img[i], i)

        # wave lengths
        waves = WAVES[:self.n_bands]
        
        # transform for (img,lbl) pairs
        try:
            img, lbl = self.transform((img, lbl))                
        except:
            raise ValueError(f'Error in transforming data with key={key}, {a}.')

        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'nyc',
                KEY_WAVE: waves,
                KEY_WEAKLBL: False}
    
    def __len__(self):
        return self.length