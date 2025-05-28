# import os 
# os.chdir('ProxyCLIP')
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from proxyclip_segmentor import ProxyCLIPSegmentation

from dataset.datasets_settings import datasets_settings_test
from dataset import DWOSMDataset, OpenEarthMapDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset

KEY_IMG = 'image'
KEY_LBL = 'label'
MASK_CODE = 255
dataset_name = 'potsdam'
cls_file = 'cls_potsdam'

def calculate_acc_from_all(pred, gt, n_classes):
    ious = []
    for i in range(1,n_classes+1):
        pred_ = (pred == i).astype(float)   
        gt_ = (gt == i).astype(float)
        intersection = np.sum(pred_ * gt_)
        union = np.sum(pred_) + np.sum(gt_) - intersection
        iou = intersection / union
        ious.append(iou)
    ious.append(np.mean(ious))
    tmp = pred==gt
    oa = np.sum(tmp[gt>0])/np.sum(gt>0)
    return ious, oa

# load data
test_dataset = PotsdamDataset('data/weakFT/test/potsdam_p512_val.lmdb',
                              n_bands=datasets_settings_test[dataset_name]['n_channels'],)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8,
                        pin_memory=True, shuffle=False, drop_last=False)
n_batch = len(test_loader)
n_ts = len(test_dataset)
print(f'Total test data: {n_ts} | Number of batches: {n_batch}')

model = ProxyCLIPSegmentation(clip_type='openai', model_type='ViT-B/16', vfm_model='dino',
                              name_path=f'ProxyCLIP/configs/{cls_file}.txt',
                              slide_stride=512, slide_crop=0,
                            #   slide_stride=112, slide_crop=336,
                              )

t0 = datetime.datetime.now().replace(microsecond=0)
preds, ys, mid_feats = [], [], []
for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
    x = batch[KEY_IMG].contiguous().to('cuda')
    y = batch[KEY_LBL].contiguous()
    seg_pred, mid_feat = model.predict(x, data_samples=None)

    preds.append(seg_pred.cpu().numpy())
    ys.append(y.numpy())
    mid_feats.append(mid_feat.cpu().numpy())
    if i==20: break
t1 = datetime.datetime.now().replace(microsecond=0)

# text embeddings
txt_embed = model.query_features.cpu().numpy()
np.save('txt_embed.npy', txt_embed) 

# accuracy
try:
    ys = np.concatenate(ys, axis=0)
    ys = np.where(ys!=MASK_CODE, ys+1, 0)
    preds = np.concatenate(preds, axis=0)+1
    n_classes = np.max(ys)
    preds[ys==0] = 0
    ts_iou_all, ts_oa = calculate_acc_from_all(preds, ys, n_classes)
    accs = np.array(ts_iou_all+[ts_oa])
    print(accs)
    # np.save('preds.npy', preds[:10])
    # np.save('ys.npy', ys[:10])
except Exception as e:
    print(ys.shape, ys.dtype, preds.shape, preds.dtype)
    print(e)

t2 = datetime.datetime.now().replace(microsecond=0)
print(f'Inference time: {t1-t0} | Acc calculating time: {t2-t1}')

np.save('clip_feat.npy', mid_feats) 
for sid in [10,15,19]:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(preds[sid], cmap='tab10', vmin=0, vmax=6, interpolation='none')
    plt.subplot(1, 2, 2)
    plt.imshow(ys[sid], cmap='tab10', vmin=0, vmax=6, interpolation='none')
    plt.savefig(f'ProxyCLIP/images/potsdam_new_{sid}.png', bbox_inches='tight')

    
    


