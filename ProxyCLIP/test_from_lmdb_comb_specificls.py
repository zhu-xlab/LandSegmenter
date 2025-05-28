import os 
# print(os.getcwd())
# os.chdir('ProxyCLIP')
import datetime
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import torch
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
    # for the separate auxiliary decoder setting
    if args.use_sep_aux_de_weak:
        args.use_aux_decoder = 8
    
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
        # trans_ts = None

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
        print(f'Test dataset: {args.dataset_name} | Number of samples: {len(test_dataset)} | Patch size: {test_dataset[0][KEY_IMG].shape[-1]}')
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
        # pred_fname = f'pred_OV_EP{ep}_{args.dataset_name}.npy'
        pred_fname = f'pmaps_OV_EP{ep}_{args.dataset_name}.h5'
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
        preds, ys, preds_clip, preds_weak = [], [], [], []
        # max_thr = 0.6
        # show_imgs = []
        for i, batch in enumerate(test_loader):
            # if i==2:
            x = batch[KEY_IMG].contiguous().to('cuda')
            y = batch[KEY_LBL].contiguous()
            clip_prob = model.predict(x, batch, return_prob=True)
            weak_prob = model.get_weak_model_prediction(batch, return_prob=True)
            # # # # # # uneven combination
            # seg_pred = (clip_prob+weak_prob)/2
            # weak_prob_max = weak_prob.max(-1)[0].max(-1)[0][:,:,None,None]
            # mask_clip = torch.ones_like(weak_prob_max)*2
            # mask_weak = torch.ones_like(weak_prob_max)*2
            # mask_clip[weak_prob_max<max_thr] = 3
            # mask_weak[weak_prob_max<max_thr] = 1
            ##
            weak_prob_max = weak_prob.max(-1)[0].max(-1)[0][:,:,None,None]
            clip_prob_max = clip_prob.max(-1)[0].max(-1)[0][:,:,None,None]
            mask_clip = torch.ones_like(weak_prob_max)*2
            mask_weak = torch.ones_like(weak_prob_max)*2
            select = (weak_prob_max<args.max_thr)*(clip_prob_max>=args.max_thr)
            mask_clip[select] = 3
            mask_weak[select] = 1
            select = (weak_prob_max>=args.max_thr)*(clip_prob_max<args.max_thr)
            mask_clip[select] = 1
            mask_weak[select] = 3
            ##
            seg_pred = mask_clip*clip_prob+mask_weak*weak_prob
            seg_pred = seg_pred/seg_pred.sum(dim=1, keepdim=True)
            # # # # # # uneven combination
            seg_pred = seg_pred.argmax(1)
            seg_clip = clip_prob.argmax(1)
            seg_weak = weak_prob.argmax(1)
            preds.append(seg_pred.cpu().numpy())
            preds_clip.append(seg_clip.cpu().numpy())
            preds_weak.append(seg_weak.cpu().numpy())
            ys.append(y.numpy())
            # if i==5: break
            # if i==0:
            # with h5py.File(test_save_path, 'w') as f:
            #     f.create_dataset('clip_prob', data=clip_prob.cpu().numpy())
            #     f.create_dataset('weak_prob', data=weak_prob.cpu().numpy())
            # break
        t1 = datetime.datetime.now().replace(microsecond=0)

        # accuracy
        try:
            ys = np.concatenate(ys, axis=0)
            ys = np.where(ys!=MASK_CODE, ys+1, 0)
            preds = np.concatenate(preds, axis=0)+1
            preds_clip = np.concatenate(preds_clip, axis=0)+1
            preds_weak = np.concatenate(preds_weak, axis=0)+1
            # with h5py.File(test_save_path, 'w') as f:
            #     f.create_dataset('preds', data=preds, dtype=np.uint8)
            #     f.create_dataset('preds_clip', data=preds_clip, dtype=np.uint8)
            #     f.create_dataset('preds_weak', data=preds_weak, dtype=np.uint8)
            #     f.create_dataset('ys', data=ys, dtype=np.uint8)
            n_classes = np.max(ys)
            preds[ys==0] = 0
            preds_clip[ys==0] = 0
            preds_weak[ys==0] = 0
            accs = []
            for p in [preds_clip, preds_weak, preds]:
                ts_iou_all, ts_oa = calculate_acc_from_all(p, ys, n_classes)
                accs_ = np.array(ts_iou_all+[ts_oa])
                accs.append(accs_)
            accs = np.array(accs)
            for acc, name in zip(accs, ['clip', 'weak', 'ovc ']):
                acc_print = np.round(acc, 4)
                print(f'{args.dataset_name} | {name} | {acc_print}')
            # print(dataset_name, accs)
            # np.save(test_save_path, accs)
            # np.save('preds.npy', preds)
            # np.save('preds_clip.npy', preds_clip)
            # np.save('preds_weak.npy', preds_weak)
            # np.save('ys.npy', ys)
            # with h5py.File(test_save_path, 'w') as f:
            #     f.create_dataset('preds', data=preds, dtype=np.uint8)
            #     f.create_dataset('preds_clip', data=preds_clip, dtype=np.uint8)
            #     f.create_dataset('preds_weak', data=preds_weak, dtype=np.uint8)
            #     f.create_dataset('ys', data=ys, dtype=np.uint8)

        except Exception as e:
            raise NotImplementedError(e)

        t2 = datetime.datetime.now().replace(microsecond=0)
        print(f'Inference time: {t1-t0} | Acc calculating time: {t2-t1}')

        # for sid in [0,10]:
        #     plt.figure()
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(preds[sid], cmap='tab10', vmin=0, vmax=6, interpolation='none')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(ys[sid], cmap='tab10', vmin=0, vmax=6, interpolation='none')
        #     plt.savefig(os.path.join(args.pdir, f'Image_OVC_{args.weak_feat_type}_EP{ep}_{args.dataset_name}_{sid}_{args.clip_slide_stride}_{args.clip_slide_crop}.png'), bbox_inches='tight')
        # for sid in [42, 60, 78, 80, 101, 106, 123, 124, 137, 140, 142, 145, 148, 150, 162, 169, 183, 191, 193]:
        #     plt.figure()
        #     plt.subplot(1, 4, 1)
        #     plt.imshow(preds_clip[sid], cmap='tab10', vmin=0, vmax=n_classes, interpolation='none')
        #     plt.axis('off')
        #     plt.subplot(1, 4, 2)
        #     plt.imshow(preds_weak[sid], cmap='tab10', vmin=0, vmax=n_classes, interpolation='none')
        #     plt.axis('off')
        #     plt.subplot(1, 4, 3)
        #     plt.imshow(preds[sid], cmap='tab10', vmin=0, vmax=n_classes, interpolation='none')
        #     plt.axis('off')
        #     plt.subplot(1, 4, 4)
        #     plt.imshow(ys[sid], cmap='tab10', vmin=0, vmax=n_classes, interpolation='none')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(args.pdir, f'Image_OVC_{args.weak_feat_type}_EP{ep}_{args.dataset_name}_{sid}_{args.clip_slide_stride}_{args.clip_slide_crop}.png'), bbox_inches='tight')

if __name__ == '__main__': 
    main()
    


