import os
import sys
import cv2
import numpy as np
import torch
import argparse

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# dataset
from dataset import DWOSMDataset, SegAugTransforms
import albumentations as A
from torch.utils.data import DataLoader

# model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


KEY_IMG = 'image'
KEY_LBL = 'label'
# KEY_CLSN = 'cn_dict'
KEY_LBLTYPE = 'label_type'
MASK_CODE = 255
INPUT_SIZE = 256
THRES = 0

CLS_DICT = {0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub and Scrub', 6:'Built area', 7:'Bare land', 8:'Ice and Snow', 9:'Clouds'}


def get_args():
    parser = argparse.ArgumentParser(description='DWOSM dataset')
    parser.add_argument('--checkpoint', dest='ckpt', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--test_path', type=str, default='/p/project1/hai_dm4eo/liu_ch/data/SSL4EO-S2/downstream_lmdb_small_remove/dw_test_lmdb')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=50, help='Batch size for test and validation sets')
    parser.add_argument('--input_type', dest='it', type=str, default='s2', choices=['s1', 's2', 's12'], help='Type of input data')
    parser.add_argument('--n_channels', type=int, default=3, help='Number of input channels.') 
    parser.add_argument('--patch_size', type=int, default=256, help='Random cropped Patch size before resizing to 1024.')
    parser.add_argument('--config_file', type=str, default='sam2_hiera_l.yaml')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--save', action='store_true', default=False, help='Save the predictions')

    args = parser.parse_args()
    return args


def calculate_iou(pred, gt, n_classes):
    ious = []
    for i in range(n_classes):
        pred_ = (pred == i).astype(float)   
        gt_ = (gt == i).astype(float)
        intersection = np.sum(pred_ * gt_)
        union = np.sum(pred_) + np.sum(gt_) - intersection
        iou = intersection / union
        ious.append(iou)
    return ious

def main():
    args = get_args()

    # devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # datasets
    trans_ts = A.Compose([A.CenterCrop(width=INPUT_SIZE, height=INPUT_SIZE)])
    def get_dataset(data_path, trans=None):
        dataset = DWOSMDataset(data_path, mode=args.it, n_bands=args.n_channels, 
                               transform=trans)
        return dataset
    # test set
    test_dataset = get_dataset(args.test_path, SegAugTransforms(trans_ts))
    # test_dataset2 = get_dataset(args.test_path)
    # dataloader: no matther what devices is used, here directly utilize DataLoader since pl will take of DDP cases
    def get_dataloader(dataset, batch_size, shuffle):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=args.num_workers, drop_last=False)
        return data_loader
    test_loader = get_dataloader(test_dataset, args.tbs, False)
    # test_loader2 = get_dataloader(test_dataset2, args.tbs, False)
    nc_total = len(CLS_DICT)
    texts = [CLS_DICT[i] for i in range(nc_total)]

    # model
    sam2 = build_sam2(args.config_file, args.ckpt, test_mode=True, apply_postprocessing=False, device=device)
    predictor = SAM2ImagePredictor(sam2)  

    # predict
    pmasks, pious, gts = [], [], []
    for i, data in enumerate(test_loader):  
        image = data[KEY_IMG].to(device)
        gt = data[KEY_LBL].numpy()
        gts.append(gt)
        predictor.set_image_batch(image)
        pred_masks, pred_ious = predictor.predict_batch(texts)
        pmasks.append(pred_masks)
        pious.append(pred_ious)
        if i==0:
            print(f'Predicted masks shape: {pred_masks.shape}')
            print(f'Predicted ious shape: {pred_ious.shape}')
            print(f'Ground truth shape: {gt.shape}')
    pmasks = np.concatenate(pmasks, axis=0)
    pious = np.concatenate(pious, axis=0)
    gts = np.concatenate(gts, axis=0)

    # evaluate
    masks = np.argmax(pmasks, axis=1)
    masks_conf = masks.copy()
    vmax = np.max(pious, axis=1)
    masks_conf[vmax < THRES] = MASK_CODE
    ious = calculate_iou(masks, gts, nc_total)
    ious_conf = calculate_iou(masks_conf, gts, nc_total)
    # for i, iou in enumerate(ious):
    #     print(f'{texts[i]}: {iou:.4f}')
    ious.append(np.mean(ious))
    ious_conf.append(np.mean(ious_conf))
    # print(f'Mean IoU: {ious[-1]:.4f}')
    print(f'IoU: {ious}')
    print(f'IoU of confidence: {ious_conf}')
    ious_all = np.array([ious, ious_conf])

    # save
    pdir = args.ckpt.split('checkpoints')[0][:-1]
    ep = args.ckpt.split('epoch=')[-1].split('-')[0]
    pdir = os.path.join(pdir, 'predictions')
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, f'ious_ep{ep}.npy'), ious_all)
    if args.save:
        np.save(os.path.join(pdir, f'pmasks_ep{ep}.npy'), pmasks)
        np.save(os.path.join(pdir, f'pious_ep{ep}.npy'), pious)
    print(f'Predictions are saved to {pdir}.')


if __name__ == '__main__':
    main()



