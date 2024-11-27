import lmdb
import pickle
import numpy as np
from torch.utils.data import Dataset

# from albumentations.pytorch import ToTensorV2
from dataset.data_properties_sentinel import SegAugTransforms

# order: NIR-R-G
# from ImageNet stats
MEANS = [131.669, 123.675, 116.28]  
STDS = [50.583, 58.395, 57.12] # from ImageNet stats 

# from current dataset
MEANS = [120.380, 81.691, 81.153]
STDS = [54.817, 39.229, 37.898]

WAVES = np.array([0.83, 0.665, 0.56])  # NIR-R-G

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


# classes:  (0, 255, 255): 1,    # Low vegetation
#           (255, 255, 255): 2,  # Impervious surfaces
#           (255, 255, 0): 3,    # Car
#           (0, 0, 255): 4,      # Roof
#           (0, 255, 0): 5       # Tree
class VaihingenDataset(Dataset):
    def __init__(self, lmdb_file, normalize=True, transform=None):
        self.lmdb_file = lmdb_file
        self.normalize = normalize
        self.n_bands = 3
        
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
        img = img[[1,2,0]]  # reorder to R-G-NIR

        # wave lengths
        waves = WAVES[[1,2,0]]
        
        # transform for (img,lbl) pairs
        try:
            img, lbl = self.transform((img, lbl))                
        except:
            raise ValueError(f'Error in transforming data with key={key}, {a}.')

        return {KEY_IMG: img, 
                KEY_LBL: lbl.long(), 
                KEY_DATASAT: 'vaihingen',
                KEY_WAVE: waves,
                KEY_WEAKLBL: False}
    
    def __len__(self):
        return self.length