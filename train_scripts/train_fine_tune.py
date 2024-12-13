import datetime
import os
import sys
import numpy as np

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader

# import lightening
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from train_scripts.get_args import get_args_fine_tune as get_args
from sam2.sam2_image_trainer_fine_tune import SAM2ImageTrainerFT

# dataset
from dataset.datasets_settings import datasets_settings_test
from dataset import DWOSMDataset, OpenEarthMapDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset
import albumentations as A



# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)

    ###### 1 - parameter setting ######
    args = get_args()
    pl.seed_everything(args.seed)

    t1 = datetime.datetime.now().replace(microsecond=0)
    patch_size = datasets_settings_test[args.dataset_name]['input_size']
    
    ###### 2 - load data ######
    # transforms
    if args.dataset_name == 'nyc':
        trans_tr = A.Compose([A.RandomCrop(width=patch_size, height=patch_size),
                              A.D4(p=0.5), # A.Flip(p=0.5),
                              A.Rotate(p=0.2),
                             ])
        trans_ts = A.Compose([A.CenterCrop(width=patch_size, height=patch_size,)])
    else:
        trans_tr = A.Compose([A.D4(p=0.5), # A.Flip(p=0.5),
                              A.Rotate(p=0.2),
                             ])
        trans_ts = None 

    # datasets
    if args.mute_adaptor:
        n_channels = 3
    else:
        n_channels = datasets_settings_test[args.dataset_name]['n_channels']
        # if 'vit' in args.dofa_type:
        #     n_channels = datasets_settings_test[args.dataset_name]['n_channels']
        # else:
        #     n_channels = 3 
        
    if args.dataset_name == 'dw':
        train_dataset = DWOSMDataset(datasets_settings_test['dw']['data_path_ft'],
                                     mode=datasets_settings_test['dw']['mode'],
                                     n_bands=n_channels,
                                     transform=trans_tr)
        test_dataset = DWOSMDataset(datasets_settings_test['dw']['data_path'],
                                    mode=datasets_settings_test['dw']['mode'],
                                    n_bands=n_channels,
                                    transform=trans_ts)
    elif args.dataset_name == 'openearthmap':
        test_dataset = OpenEarthMapDataset(datasets_settings_test['openearthmap']['data_path'],
                                        transform=trans_ts)
    elif args.dataset_name == 'multisenge':
        train_dataset = MultiSenGeDataset(datasets_settings_test['multisenge']['data_path_ft'],
                                          n_bands=n_channels,
                                          transform=trans_tr)
        test_dataset = MultiSenGeDataset(datasets_settings_test['multisenge']['data_path'],
                                         n_bands=n_channels,
                                         transform=trans_ts)
    elif args.dataset_name == 'osm':
        train_dataset = DWOSMDataset(datasets_settings_test['osm']['data_path_ft'],
                                    mode=datasets_settings_test['osm']['mode'],
                                    n_bands=n_channels,
                                    lbl_type='osm',
                                    transform=trans_tr)
        test_dataset = DWOSMDataset(datasets_settings_test['osm']['data_path'],
                                    mode=datasets_settings_test['osm']['mode'],
                                    n_bands=n_channels,
                                    lbl_type='osm',
                                    transform=trans_ts)
    elif args.dataset_name == 'vaihingen':
        train_dataset = VaihingenDataset(datasets_settings_test['vaihingen']['data_path_ft'],
                                         transform=trans_tr)
        test_dataset = VaihingenDataset(datasets_settings_test['vaihingen']['data_path'],
                                        transform=trans_ts)
    elif args.dataset_name == 'nyc':
        train_dataset = NYCDataset(datasets_settings_test['nyc']['data_path_ft'],
                                   n_bands=n_channels,
                                   transform=trans_tr)
        test_dataset = NYCDataset(datasets_settings_test['nyc']['data_path'],
                                  n_bands=n_channels,
                                  transform=trans_ts)
    elif args.dataset_name == 'loveda':
        train_dataset = LoveDADataset(datasets_settings_test['loveda']['data_path_ft'],
                                      transform=trans_tr)
        test_dataset = LoveDADataset(datasets_settings_test['loveda']['data_path'],)
    elif args.dataset_name == 'potsdam':
        train_dataset = PotsdamDataset(datasets_settings_test['potsdam']['data_path_ft'],
                                        n_bands=n_channels,
                                        transform=trans_tr)
        test_dataset = PotsdamDataset(datasets_settings_test[args.dataset_name]['data_path'],
                                      n_bands=n_channels,)
    print(f'Dataset: {args.dataset_name} | #training: {len(train_dataset)} | #test: {len(test_dataset)} | Patch size: {patch_size}')
    
    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, num_workers=args.num_workers,
                              pin_memory=True, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.tbs, num_workers=args.num_workers,
                             pin_memory=True, shuffle=False, drop_last=False)
    args.n_batch_tr = len(train_loader)
    args.n_ts = len(test_dataset)
    # args.n_classes = len(class_names[args.dataset_name])

    # saving path
    if args.ft_wpath is None or args.ft_wpath=='None':
        if args.mute_adaptor:
            args.gname = 'sam2_ft'
        else:
            if args.dofa_type is None or args.dofa_type=='None':
                args.gname = 'sam2_adaptor_ft'
            else:
                args.gname = f'sam2_adaptor_ft_{args.dofa_type}'
            # raise ValueError('Please provide the path to the pre-trained model for fine-tuning or mute the adaptor!')
        args.pdir = 'Results_' + args.gname
    else:
        args.pdir = args.ft_wpath.split('checkpoints')[0]
        gname_list = args.pdir.split('/')[-4:-2]
        args.gname = args.pdir.split('/')[-5][8:]
        args.gname = args.gname + '_' + '_'.join(gname_list)
        resume_epoch = args.ft_wpath.split("epoch=")[-1].split("-")[0]
        args.gname += f'_fe{resume_epoch}'
        args.pdir = os.path.join(args.pdir, f'fine_tune_fe{resume_epoch}')
    # lr
    ea = int(np.ceil(np.log10(1/args.lr)))
    eb = int(np.round(args.lr*10**ea))
    bce_tag = 'b' if args.use_binary_loss else ''
    bce_tag += 'nc' if args.use_class_multiplier else ''
    freeze_tag = 'FBO_' if args.freeze_backbones_only else ''
    args.job_type = f'{freeze_tag}EP{args.epochs}{bce_tag}_{args.dataset_name}_lr{eb}e{ea}' # f'{args.eft_tag}{args.eft_ep}_{feb}e{fea}_{args.dft_tag}_lr{eb}e{ea}'
    # optimizer
    if args.schdt!='none': 
        args.job_type += f'_{args.schdt}'
        if args.schdt=='step':
            for dv in args.schdv: args.job_type += f'_{int(dv)}'
        elif args.schdt=='exp':
            args.job_type += f'_{int(args.schdv[0]*100)}'
        elif args.schdt=='cos':
            if args.schdv[0]>0:
                args.job_type += f'_w{int(args.schdv[0])}_{int(args.schdv[1])}'
            else:
                args.job_type += f'_{int(args.schdv[1])}'
            if len(args.schdv)>2: 
                ea = int(np.ceil(np.log10(1/args.schdv[2])))
                eb = int(args.schdv[2]*10**ea)
                args.job_type += f'_f{eb}e{ea}'  # final_lr setting
    args.pdir = os.path.join(args.pdir, args.job_type)
    checkpoints_dir = os.path.join(args.pdir, 'checkpoints')
    # create directory
    if not os.path.exists(args.pdir): 
        try:
            os.makedirs(args.pdir)
        except Exception as inner_e:
            assert os.path.exists(args.pdir), f'Fine-tuning directory does not exist and cannot be created: {args.pdir}. Error: {inner_e}'
    pred_fname = 'evaluation_results.npy'
    args.save_path = os.path.join(args.pdir, pred_fname)
    # setting logger
    os.environ["WANDB_MODE"] = args.wmode
    wandb_logger = WandbLogger(project=args.pjn, entity=args.entity, 
                               group=args.gname, job_type=args.job_type,
                               name=f's{args.seed}_b{args.bs}',
                               save_dir=args.pdir)
    

    ###### 3 - define the model ######
    torch.set_float32_matmul_precision('high')
    # if args.use_sep_aux_de_weak:
    #     args.dataset_name = ['dynamicearthnet', 'openearthmap', 'sbtn', 'worldcover', 'iran', 'ghsl', 'nlcd', 'usfs']
    # build model
    model = SAM2ImageTrainerFT(args)
    
    
    ###### 4 - train the model ######
    print('Start fine-tuning...')
    callbacks = [ModelCheckpoint(dirpath = checkpoints_dir, every_n_epochs = args.save_inv, 
                                #  save_weights_only=True, filename='model_{epoch:02d}_{val_loss:.2f}',
                                 save_top_k=-1,  # Save all models
                                 save_last=False
                                 )]
    if args.schdt!='none': callbacks += [LearningRateMonitor(logging_interval='epoch')]
    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         strategy=args.strategy,
                         num_nodes=args.nodes,
                         precision='16-mixed' if args.amp else '32-true',
                         max_epochs=args.epochs, 
                         check_val_every_n_epoch=args.display_inv,
                         logger=wandb_logger,
                         enable_progress_bar=False,
                         callbacks=callbacks,
                         use_distributed_sampler=(args.ntasks <= 1),
                         )    
    
    trainer.fit(model, train_loader, test_loader)

    t3 = datetime.datetime.now().replace(microsecond=0)
    print(f'Fine-tuning is finished|Total spent time:{t3-t0}!')
    
    
    
if __name__ == '__main__':
    main()


