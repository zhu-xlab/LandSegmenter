import os 
# print(os.getcwd())
# os.chdir('ProxyCLIP')
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from proxyclip_segmentor import ProxyCLIPSegmentation

import albumentations as A
from dataset.datasets_settings import datasets_settings_test
from dataset import DWOSMDataset, OpenEarthMapDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset
from sam2.get_args import get_args_ov as get_args


KEY_IMG = 'image'
KEY_LBL = 'label'
MASK_CODE = 255
# dataset_name = 'potsdam'
# cls_file = 'cls_potsdam'

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

def main():
    ###### 1 - parameter setting ######
    global args
    args = get_args()
    assert args.test_only, 'This script is for testing only!'
    if len(args.test_batch_sizes) == 1:
        args.test_batch_sizes = args.test_batch_sizes * len(args.test_dataset_names)
    assert len(args.test_batch_sizes) == len(args.test_dataset_names), 'The number of test batch sizes should be 1 or the same as the number of test datasets!'
    
    for di, args.dataset_name in enumerate(args.test_dataset_names):
        t1 = datetime.datetime.now().replace(microsecond=0)
        patch_size = datasets_settings_test[args.dataset_name]['input_size']
        args.tbs = args.test_batch_sizes[di]
        
        ###### 2 - load data ######
        # datasets
        if args.dataset_name == 'nyc':
            trans_ts = A.Compose([A.CenterCrop(width=patch_size, height=patch_size,)])
        else:
            trans_ts = None 

        # test data
        if args.dataset_name == 'dw':
            test_dataset = DWOSMDataset(datasets_settings_test['dw']['data_path'],
                                        mode=datasets_settings_test['dw']['mode'],
                                        n_bands=datasets_settings_test['dw']['n_channels'],
                                        transform=trans_ts)
        elif args.dataset_name == 'openearthmap':
            test_dataset = OpenEarthMapDataset(datasets_settings_test['openearthmap']['data_path'],
                                            transform=trans_ts)
        elif args.dataset_name == 'multisenge':
            test_dataset = MultiSenGeDataset(datasets_settings_test['multisenge']['data_path'],
                                            transform=trans_ts)
        elif args.dataset_name == 'osm':
            test_dataset = DWOSMDataset(datasets_settings_test['osm']['data_path'],
                                        mode=datasets_settings_test['osm']['mode'],
                                        n_bands=datasets_settings_test['osm']['n_channels'],
                                        lbl_type='osm')
        elif args.dataset_name == 'vaihingen':
            test_dataset = VaihingenDataset(datasets_settings_test['vaihingen']['data_path'],
                                            transform=trans_ts)
        elif args.dataset_name == 'nyc':
            test_dataset = NYCDataset(datasets_settings_test['nyc']['data_path'],
                                    n_bands=datasets_settings_test['nyc']['n_channels'],
                                    transform=trans_ts)
        elif args.dataset_name == 'loveda':
            test_dataset = LoveDADataset(datasets_settings_test['loveda']['data_path'],)
        elif args.dataset_name == 'potsdam':
            test_dataset = PotsdamDataset(datasets_settings_test[args.dataset_name]['data_path'],
                                        n_bands=datasets_settings_test[args.dataset_name]['n_channels'],)
        print(f'Test dataset: {args.dataset_name} | Number of samples: {len(test_dataset)} | Patch size: {patch_size}')
        test_loader = DataLoader(test_dataset, batch_size=args.tbs, num_workers=args.num_workers,
                                pin_memory=True, shuffle=False, drop_last=False)
        args.n_batch = len(test_loader)
        args.n_ts = len(test_dataset)
        # args.n_classes = len(class_names[args.dataset_name])

        # saving path
        args.pdir = args.ts_wpath.split('checkpoints')[0]
        if args.test_use_multiple_prompts:
            args.pdir = os.path.join(args.pdir, 'accuracy_mp')
        else:
            args.pdir = os.path.join(args.pdir, 'accuracy')
        if not os.path.exists(args.pdir): 
            try:
                os.makedirs(args.pdir)
            except Exception as inner_e:
                assert os.path.exists(args.pdir), f'Accuracy directory does not exist and cannot be created: {args.pdir}. Error: {inner_e}'
        ep = args.ts_wpath.split("epoch=")[-1].split("-")[0]
        pred_fname = f'pred_OV_EP{ep}_{args.dataset_name}.npy'
        test_save_path = os.path.join(args.pdir, pred_fname)
        
    # # load data
    # test_dataset = PotsdamDataset('data/weakFT/test/potsdam_p512_val.lmdb',
    #                             n_bands=datasets_settings_test[dataset_name]['n_channels'],)
    # test_loader = DataLoader(test_dataset, batch_size=1, num_workers=8,
    #                         pin_memory=True, shuffle=False, drop_last=False)
    # n_batch = len(test_loader)
    # n_ts = len(test_dataset)
    # print(f'Total test data: {n_ts} | Number of batches: {n_batch}')

        model = ProxyCLIPSegmentation(clip_type=args.clip_type, 
                                      model_type=args.clip_visual_type, 
                                      vfm_model=args.vfm_model,
                                      name_path=f'ProxyCLIP/configs/cls_{args.dataset_name}.txt',
                                      slide_stride=args.clip_slide_stride, slide_crop=args.clip_slide_crop,
                                      #   slide_stride=112, slide_crop=336,
                                      args=args,
                                      )
        print(f'clip_slide_stride:{args.clip_slide_stride}, clip_slide_crop:{args.clip_slide_crop}')

        # predictions
        t0 = datetime.datetime.now().replace(microsecond=0)
        preds, ys = [], []
        for i, batch in enumerate(test_loader):
            x = batch[KEY_IMG].contiguous().to('cuda')
            y = batch[KEY_LBL].contiguous()
            seg_pred = model.predict(x, batch)

            preds.append(seg_pred.cpu().numpy())
            ys.append(y.numpy())
            if i==5: break
        t1 = datetime.datetime.now().replace(microsecond=0)

        # accuracy
        try:
            ys = np.concatenate(ys, axis=0)
            ys = np.where(ys!=MASK_CODE, ys+1, 0)
            preds = np.concatenate(preds, axis=0)+1
            n_classes = np.max(ys)
            preds[ys==0] = 0
            ts_iou_all, ts_oa = calculate_acc_from_all(preds, ys, n_classes)
            accs = np.array(ts_iou_all+[ts_oa])
            print(args.dataset_name, accs)
            np.save(test_save_path, accs)
            # np.save('preds.npy', preds[:10])
            # np.save('ys.npy', ys[:10])
        except Exception as e:
            print(ys.shape, ys.dtype, preds.shape, preds.dtype)
            raise NotImplementedError(e)

        t2 = datetime.datetime.now().replace(microsecond=0)
        print(f'Inference time: {t1-t0} | Acc calculating time: {t2-t1}')

        for sid in [0,10]:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(preds[sid], cmap='tab10', vmin=0, vmax=6, interpolation='none')
            plt.subplot(1, 2, 2)
            plt.imshow(ys[sid], cmap='tab10', vmin=0, vmax=6, interpolation='none')
            plt.savefig(os.path.join(args.pdir, f'Image_OV_{args.weak_feat_type}_EP{ep}_{args.dataset_name}_{sid}_{args.clip_slide_stride}_{args.clip_slide_crop}.png'), bbox_inches='tight')


if __name__ == '__main__': 
    main()
    


