import datetime
import os
import sys
import h5py
import numpy as np

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# import lightening
import pytorch_lightning as pl
from sam2.get_args import get_args_test_framework as get_args
from sam2.model_feature_generation import GenerateFeat

# dataset
from dataset.datasets_settings import datasets_settings_test, class_names
from dataset import DWOSMDataset, OpenEarthMapDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset
import albumentations as A


N = 60

def claculate_entropy(probs, dim=-1):
    """
    Calculate entropy for a given probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor. Assumes values sum to 1 along `dim`.
        dim (int): Dimension along which to calculate entropy.

    Returns:
        torch.Tensor: The entropy of the probability distribution.
    """
    # Avoid log(0) by adding a small epsilon
    eps = 1e-9
    probs = probs.clamp(min=eps)  # Ensure probabilities are non-zero
    return -torch.sum(probs * torch.log(probs), dim=dim)

def main():
    t0 = datetime.datetime.now().replace(microsecond=0)

    # args
    global args
    args = get_args()
    assert args.test_only, 'This script is for testing only!'
    if len(args.test_batch_sizes) == 1:
        args.test_batch_sizes = args.test_batch_sizes * len(args.test_dataset_names)
    assert len(args.test_batch_sizes) == len(args.test_dataset_names), 'The number of test batch sizes should be 1 or the same as the number of test datasets!'

    # dataloaders
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
        # print(f'Test dataset: {args.dataset_name} | Number of samples: {len(test_dataset)} | Patch size: {patch_size}')
        tbs = N if N <= args.tbs else args.tbs
        test_loader = DataLoader(test_dataset, batch_size=tbs, num_workers=args.num_workers,
                                pin_memory=False, shuffle=False, drop_last=False)
        n_batch = np.ceil(N/tbs).astype(int)
        # args.n_classes = len(class_names[args.dataset_name])

        # saving path
        args.pdir = args.ts_wpath.split('checkpoints')[0]
        args.pdir = os.path.join(args.pdir, 'features')
        if not os.path.exists(args.pdir): 
            try:
                os.makedirs(args.pdir)
            except Exception as inner_e:
                assert os.path.exists(args.pdir), f'Feature directory does not exist and cannot be created: {args.pdir}. Error: {inner_e}'
        ep = args.ts_wpath.split("epoch=")[-1].split("-")[0]
        pred_fname = f'unc_EP{ep}_{args.dataset_name}_N{N}.h5'
        test_save_path = os.path.join(args.pdir, pred_fname)
        

        ###### 3 - define the model ######
        torch.set_float32_matmul_precision('high')
        dataset_name = args.dataset_name
        if args.use_sep_aux_de_weak:
            args.dataset_name = ['dynamicearthnet', 'openearthmap', 'sbtn', 'worldcover', 'iran', 'ghsl', 'nlcd', 'usfs']
        # build model
        model = GenerateFeat(args)

        # class list
        cls_dict = class_names[args.dataset_name]
        class_id_list = []
        for k in cls_dict:
            class_id_list = class_id_list + [k]*len(cls_dict[k])
        class_id_list = torch.Tensor(class_id_list).to(torch.int64).to('cuda')
        
        ###### 4 - generate features ######
        logits_all, entropy_all, entropy_norm_all = [], [], []
        with autocast():
            for i, batch in enumerate(test_loader):
                # inds = batch['ind']
                # print(inds)
                outs_dict = model.predict_with_multiple_prompts(batch, return_embeddings=False)

                mask_list = outs_dict['masks_list']
                logits = mask_list[-1]

                num_cls, num_queries = max(class_id_list) + 1, len(class_id_list)
                logit_min = logits.min()
                if num_cls != num_queries:
                    logits = logits.unsqueeze(1)-logit_min
                    cls_index = torch.nn.functional.one_hot(class_id_list)
                    cls_index = cls_index.T.view(1, num_cls, num_queries, 1, 1)
                    logits = (logits * cls_index).max(2)[0]
                logits = logits + logit_min
                
                probs = logits.softmax(dim=1)
                entropy = claculate_entropy(probs, dim=1)
                entropy_norm = entropy/np.log(probs.shape[1])

                logits_all.append(logits)
                entropy_all.append(entropy)
                entropy_norm_all.append(entropy_norm)

                if i>=n_batch-1: break
        
        logits_all = torch.cat(logits_all, 0)
        entropy_all = torch.cat(entropy_all, 0)
        entropy_norm_all = torch.cat(entropy_norm_all, 0)
        with h5py.File(test_save_path, 'w') as f:
            f.create_dataset('logits', data=logits_all.cpu().numpy())
            f.create_dataset('entropy', data=entropy_all.cpu().numpy())
            f.create_dataset('entropy_norm', data=entropy_norm_all.cpu().numpy())
        
        # for i in [0,5,9]:
        #     plt.figure()
        #     plt.subplot(1,3,1)
        #     print('logits shape:', logits.shape)
        #     mask_show = logits[i].argmax(0)
        #     print('mask_show shape:', mask_show.shape, mask_show.min(), mask_show.max())
        #     plt.imshow(mask_show.cpu().numpy()+1, vmin=0, vmax=6, cmap='tab10', interpolation='none')
        #     plt.subplot(1,3,2)
        #     plt.imshow(entropy[i].cpu().numpy(), interpolation='none')
        #     plt.colorbar()
        #     plt.subplot(1,3,3)
        #     plt.imshow(entropy_norm[i].cpu().numpy(), interpolation='none')
        #     plt.colorbar()
        #     plt.savefig(os.path.join(args.pdir, f'Image_unc_EP{ep}_{args.dataset_name}_{i}.png'), bbox_inches='tight')

    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Prediction of {args.test_dataset_names} is finished, using {t1-t0}')



if __name__ == '__main__':
    main()
