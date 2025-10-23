import os 
import datetime
import openpyxl
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
from dataset.datasets_settings import datasets_settings_test, class_names
from dataset import DWOSMDataset, MultiSenGeDataset, NYCDataset, LoveDADataset, PotsdamDataset
from sam2.get_args import get_args_ov as get_args

import json

KEY_IMG = 'image'
KEY_LBL = 'label'
MASK_CODE = 255

COLUMN_INDS = {0:'D1', 1:'E1', 2:'F1', 3:'G1', 4:'H1', 5:'I1', 6:'J1', 7:'K1', 8:'L1', 9:'M1', 10:'N1', 11:'O1', 12:'P1', 13:'Q1', 14:'R1', 15:'S1', 16:'T1',
              17:'U1', 18:'V1', 19:'W1', 20:'X1', 21:'Y1', 22:'Z1'}

def append_experiment_result(file_path, result):
    try:
        workbook = openpyxl.load_workbook(file_path)
    except FileNotFoundError:
        workbook = openpyxl.Workbook()

    sheet = workbook.active

    d = 0
    if sheet['A1'].value is None:
        sheet['A1'] = 'Dataset'
        sheet['B1'] = 'Epoch'
        sheet['C1'] = 'Model'
        for i, name in enumerate(result['names']):
            sheet[COLUMN_INDS[i+d]] = name
        sheet[COLUMN_INDS[i+d+1]] = 'mIoU'
        sheet[COLUMN_INDS[i+d+2]] = 'OA'

    last_row = sheet.max_row+1

    sheet.cell(row=last_row, column=1, value=result['Dataset'])
    sheet.cell(row=last_row, column=2, value=result['Epoch'])
    sheet.cell(row=last_row, column=3, value=result['Model'])
    for i, iou in enumerate(result['IoUs']):
        sheet.cell(row=last_row, column=4+d+i, value=iou)
    sheet.cell(row=last_row, column=5+d+i, value=result['mIoU'])
    sheet.cell(row=last_row, column=6+d+i, value=result['OA'])

    workbook.save(file_path)

def get_class_name_strings(dataset_name):
    cls_dict = class_names[dataset_name]
    name_list = [cls_dict[k][0] for k in cls_dict]
    return name_list
    
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
        elif args.dataset_name == 'multisenge':
            test_dataset = MultiSenGeDataset(datasets_settings_test['multisenge']['data_path'],
                                            transform=trans_ts)
        elif args.dataset_name == 'osm':
            test_dataset = DWOSMDataset(datasets_settings_test['osm']['data_path'],
                                        mode=datasets_settings_test['osm']['mode'],
                                        n_bands=datasets_settings_test['osm']['n_channels'],
                                        lbl_type='osm')
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

        ###### 3 - set saving path ###### 
        if not os.path.exists(args.save_dir): 
            try:
                os.makedirs(args.save_dir)
            except Exception as inner_e:
                assert os.path.exists(args.save_dir), f'Accuracy directory does not exist and cannot be created: {args.save_dir}. Error: {inner_e}'
        ep = args.ts_wpath.split("epoch=")[-1].split(".")[0]
        pred_fname = f'pred_OV_{args.dataset_name}.xlsx'
        test_save_path = os.path.join(args.save_dir, pred_fname)

        ###### 4 - predict ######
        model = ProxyCLIPSegmentation(clip_type=args.clip_type, 
                                      model_type=args.clip_visual_type, 
                                      vfm_model=args.vfm_model,
                                      name_path=f'ProxyCLIP/configs/cls_{args.dataset_name}.txt',
                                      slide_stride=args.clip_slide_stride, slide_crop=args.clip_slide_crop,
                                      args=args,
                                      )

        t0 = datetime.datetime.now().replace(microsecond=0)
        preds, ys = [], []
        for i, batch in tqdm(enumerate(test_loader), total=args.n_batch):
            x = batch[KEY_IMG].contiguous().to('cuda')
            y = batch[KEY_LBL].contiguous()
            clip_prob = model.predict(x, batch, return_prob=True)
            weak_prob = model.get_weak_model_prediction(batch, return_prob=True)
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
            seg_pred = mask_clip*clip_prob+mask_weak*weak_prob
            seg_pred = seg_pred/seg_pred.sum(dim=1, keepdim=True)
            seg_pred = seg_pred.argmax(1)
            preds.append(seg_pred.cpu().numpy())
            ys.append(y.numpy())
        t1 = datetime.datetime.now().replace(microsecond=0)

        ###### 5 - calculate accuracy ###### 
        try:
            ys = np.concatenate(ys, axis=0)
            ys = np.where(ys!=MASK_CODE, ys+1, 0)
            preds = np.concatenate(preds, axis=0)+1
            n_classes = np.max(ys)
            preds[ys==0] = 0
            accs = []
            for p in [preds]:  
                ts_iou_all, ts_oa = calculate_acc_from_all(p, ys, n_classes)
                accs_ = np.array(ts_iou_all+[ts_oa])
                accs.append(accs_)
            accs = np.array(accs)
            for acc, name in zip(accs, ['Fusion']):
                acc_print = np.round(acc, 4)
                print(f'{args.dataset_name} | {name:>15} | mIoU = {acc_print[-2]} | OA = {acc_print[-1]}')

            # record results into xlsx
            for m, a in zip(['Fusion'], accs):
                results = {'Dataset': args.dataset_name,
                           'Epoch': ep,
                           'Model': m,
                           'mIoU': f'{a[-2]*100:.2f}',
                           'OA': f'{a[-1]*100:.2f}',
                           'IoUs': [f'{i*100:.2f}' for i in a[:-2]],
                           'names': get_class_name_strings(args.dataset_name),
                           }
                append_experiment_result(test_save_path, results)
        except Exception as e:
            raise NotImplementedError(e)

        t2 = datetime.datetime.now().replace(microsecond=0)
        print(f'Inference time: {t1-t0} | Acc calculating time: {t2-t1}')

if __name__ == '__main__': 
    main()
    


