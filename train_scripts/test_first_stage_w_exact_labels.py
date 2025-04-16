import datetime
import os
import sys

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader

# import lightening
import pytorch_lightning as pl
from sam2.get_args import get_args_test_framework as get_args
from sam2.sam2_image_trainer import SAM2ImageTrainer
from sam2.sam2_image_w_auxDe_trainer import SAM2ImageTrainerAux
from sam2.sam2_image_w_sepAuxDe4Weak_trainer import SAM2ImageTrainerSepAuxDe4Weak

# dataset
from dataset.datasets_settings import datasets_settings_test
from dataset import DWOSMDataset, OpenEarthMapDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset
import albumentations as A



# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)

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
        pred_fname = f'pred_EP{args.ts_wpath.split("epoch=")[-1].split("-")[0]}_{args.dataset_name}.npy'
        args.test_save_path = os.path.join(args.pdir, pred_fname)
        

        ###### 3 - define the model ######
        torch.set_float32_matmul_precision('high')
        dataset_name = args.dataset_name
        if args.use_sep_aux_de_weak:
            args.dataset_name = ['dynamicearthnet', 'openearthmap', 'sbtn', 'worldcover', 'iran', 'ghsl', 'nlcd', 'usfs']
        # build model
        if args.use_sep_aux_de_weak:
            model = SAM2ImageTrainerSepAuxDe4Weak(args)
        if args.use_aux_decoder:
            model = SAM2ImageTrainerAux(args)
        else:
            model = SAM2ImageTrainer(args)
        
        
        ###### 4 - train the model ######
        print(f'Start testing with {args.ts_wpath}')
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=args.devices,
                             strategy=args.strategy,
                             num_nodes=args.nodes,
                             precision='16-mixed' if args.amp else '32-true',
                             max_epochs=args.epochs, 
                             check_val_every_n_epoch=args.display_inv,
                             enable_progress_bar=False,
                             logger=False,
                             )    

        # test the model
        trainer.test(model=model, ckpt_path=args.ts_wpath, dataloaders=test_loader, verbose=True)
        t2 = datetime.datetime.now().replace(microsecond=0)
        print(f'Testing for {dataset_name} is finished| time:{t2-t1}!\n')
        torch.cuda.empty_cache()

    t3 = datetime.datetime.now().replace(microsecond=0)
    print(f'Testing is finished|Total spent time:{t3-t0}!\n')
    
    
    
if __name__ == '__main__':
    main()


