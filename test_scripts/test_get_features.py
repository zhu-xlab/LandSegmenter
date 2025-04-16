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
from torch.utils.data import DataLoader

# import lightening
import pytorch_lightning as pl
from train_scripts.get_args import get_args_test_framework as get_args
from model_feature_generation import GenerateFeat

# dataset
from dataset.datasets_settings import datasets_settings_test
from dataset import DWOSMDataset, OpenEarthMapDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset
import albumentations as A


N = 20

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
        pred_fname = f'feat_EP{args.ts_wpath.split("epoch=")[-1].split("-")[0]}_{args.dataset_name}_N{N}.h5'
        test_save_path = os.path.join(args.pdir, pred_fname)
        

        ###### 3 - define the model ######
        torch.set_float32_matmul_precision('high')
        dataset_name = args.dataset_name
        if args.use_sep_aux_de_weak:
            args.dataset_name = ['dynamicearthnet', 'openearthmap', 'sbtn', 'worldcover', 'iran', 'ghsl', 'nlcd', 'usfs']
        # build model
        model = GenerateFeat(args)
        
        ###### 4 - generate features ######
        for i, batch in enumerate(test_loader):
            # inds = batch['ind']
            # print(inds)
            outs_dict = model.predict_with_multiple_prompts(batch)

            with h5py.File(test_save_path, 'w') as f:
                try:
                    for k in outs_dict:
                        if k == 'masks_list': 
                            keys = ['masks'] if len(outs_dict[k])==1 else ['masks', 'masks_up']
                            for v, vk in zip(outs_dict[k], keys):
                                f.create_dataset(vk, data=v.cpu().numpy())
                        elif k == 'img_feats':
                            img_feats = outs_dict[k]
                            for ik in img_feats:
                                if type(img_feats[ik])==type([]):
                                    for ii, img_f in enumerate(img_feats[ik]):
                                        f.create_dataset(f'{ik}_{ii}', data=img_f.cpu().numpy())
                                else:
                                    f.create_dataset(ik, data=img_feats[ik].cpu().numpy())
                        elif k == 'output_embeddings':
                            keys = ['out_text_embeddings', 'out_image_embeddings', 'preout_img_embeddings']
                            for v, vk in zip(outs_dict[k], keys):
                                f.create_dataset(vk, data=v.cpu().numpy())
                        else:
                            f.create_dataset(k, data=outs_dict[k].cpu().numpy())
                except:
                    raise ValueError(f'Error key: {k}-{type(outs_dict[k])}')

            if i>=n_batch-1: break

    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Prediction of {args.test_dataset_names} is finished, using {t1-t0}')


if __name__ == '__main__':
    main()
