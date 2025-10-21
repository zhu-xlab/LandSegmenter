import datetime
import os
import sys
import pathlib
import argparse
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
from models.pl_finetune_mask import UNET_PL_FT

# dataset
from dataset.datasets_settings import datasets_settings_test, class_names
from dataset import DWOSMDataset, MultiSenGeDataset, VaihingenDataset, NYCDataset, LoveDADataset, PotsdamDataset
import albumentations as A


PATCH_SIZE = {'scalemae': 16, 'dofa': 16, 
              'satmae': 8, 'satmaepp': 8,
              'satmae-3b': 16, 'satmaepp-3b': 16,
              'mfs2': 16, 'decurs2': 16}
IMAGE_SIZE = {'scalemae': 224, 'dofa': 224, 
              'satmae': 96, 'satmaepp': 96, 
              'satmae-3b': 224, 'satmaepp-3b': 224, 
              'mfs2': 256, 'decurs2': 256}


def get_args_ft():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    # data settings
    parser.add_argument('--dataset_name', type=str, default='dw', help='Names of the datasets used for fine-tuning')
    parser.add_argument('--dataset_proportion', type=float, default=1, help='Proportion of the dataset used for fine-tuning')
    # saving directory
    parser.add_argument('--save_dir', type=str, default='Results_test')  
    # model settings
    parser.add_argument('--pretrain-model-name', type=str, dest='pmn', choices=['scalemae', 'satmae', 'satmaepp', 'dofa', 'mfs2', 'decurs2'])
    parser.add_argument('--pretrain_checkpoint_path', dest='wpath', type=str, default=None, help='Path of model weights to fine tune.')
    parser.add_argument('--model_framework', dest='frm', type=str, default='unet', choices=['upernet','segmenter','unet','deeplabv3','deeplabv3plus','pspnet','linknet','fpn',
                                                                                            'pan','manet','unetplusplus'],
                        help='Type of used model framework.')
    parser.add_argument('--model_type', dest='mt', type=str, default='resnet50', # choices=['resnet50','vit_small_patch16_224' (decur)]
                        help='Type of used models. It is mainly utilized to define the encoder part') 
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of input channels.') 
    parser.add_argument('--loss_type', type=str, default='cd',
                        help='Type of used segmentation loss function for training. ' +
                        'Specifically, \'c\' represents CrossEntropy loss, \'d\' represents Dice loss, \'f\' represents focal loss.') 
    parser.add_argument('--resume', type=str, default=False,
                        help='Path of the checkpoint file from which the training is resumed.')
    # general data settings 
    parser.add_argument('--if_mask',  action='store_true',
                        help='Indicate whether using mask strategy to train model') 
    parser.add_argument('--batch_size', dest='bs', type=int, default=5, 
                        help='Batch size')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=5, 
                        help='Batch size for test and validation sets')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, 
                        help='Number of workers for dataloaders')   
    # device settings
    parser.add_argument('--accelerator', type=str, default='cpu', choices=['cpu','gpu'], 
                        help='Type of used accelerator')
    parser.add_argument('--slurm', action='store_true', default=False, 
                        help='Whether to use slurm or not')
    # general training settings
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--optimizer', dest='opt', type=str, default='adamw', choices=['adamw', 'adam','sgd'],
                        help='Type of used optimizer.')
    parser.add_argument('--learning_rate', '-l', dest='lr', metavar='LR', type=float, default=0.001, 
                        help='Initial learning rate')
    parser.add_argument('--amp', action='store_false', 
                        help='Use mixed precision or not')
    parser.add_argument('--cal_tr_acc', dest='cal_tr', action='store_true', 
                        help='Whether to calculate training accuracies or not')
    # wandb settings
    parser.add_argument('--project_name', dest='pjn', type=str, default='NLPR_SSL4EO', 
                        help='Name of the wandb project to record statistics.')
    parser.add_argument('--entity_name', dest='entity', type=str, default='vikki023', 
                        help='Name of the entity of wandb.')
    parser.add_argument('--wandb_mode', dest='wmode', type=str, default='online', choices=["online", "offline", "disabled"],
                        help='Setting of wandb.init modes ("online", "offline" or "disabled")')
    # others
    parser.add_argument('--display_interval', dest='display_inv', type=int, default=1, 
                        help='Set the display interval during the training')
    parser.add_argument('--save_interval', dest='save_inv', type=int, default=1,
                        help='Set the save interval during the training')
    parser.add_argument("--batch_to_wandb", action="store_true", help="If true, log batch-wise training results to wandb")
    parser.add_argument('--seed', type=int, default=42)
    # finetune settings
    parser.add_argument('--weight_load_tag', dest='wload_tag', type=str, default='encoder', choices=['encoder','all','none','de1','de2','de3','de4', 'wocls'],  # 'wocls': dec4
                        help='Tag of weight loading type for recording.')
    parser.add_argument('--encoder_finetune_tag', dest='eft_tag', type=str, default='etr', choices=['efix','eft','etr'],
                        help='Tag of finetuning type of encoder for recording.')
    parser.add_argument('--decoder_finetune_tag', dest='dft_tag', type=str, default='dsam', choices=['dsam','dasc','ddes'],
                        help='Tag of finetuning type of decoder for recording.')
    parser.add_argument('--lr_varying_factor', dest='lr_vf', type=float, default=0.8,
                        help='Varying factor of learning rate when using different lrs for different layers')
    parser.add_argument('--scheduler_type', dest='schdt', type=str, default='none', choices=['none','cos','step','exp'], 
                        help='Type of scheduler to use')
    parser.add_argument('--scheduler_values', dest='schdv', type=float, nargs='*', default=[],
                        help='Values of scheduler to use (the epoch indexes for step scheduler; number of iterations for cosine scheduler)')
    parser.add_argument('--encoder_finetune_epoch', dest='eft_ep', type=int, default=0, 
                        help='Epoch index when starting to finetune the encoder if eft_tag is \'eft\'')
    # for test
    parser.add_argument('--only_test', action='store_true', default=False,
                        help='Only test models.')
    parser.add_argument('--test_weight_path', dest='ts_wpath', type=str, default=None, 
                        help='Path of test model weights.')
    
    # initialize arguments
    args = parser.parse_args()
    
    # computing environment parameters
    args = get_device_info(args)
    
    return args

 
def get_device_info(args):
    if args.slurm:
        args.slurm = "SLURM_JOB_ID" in os.environ
        
    if args.slurm:
        args.nodes = int(os.environ["SLURM_NNODES"])
        args.devices = int(os.environ["SLURM_TASKS_PER_NODE"])
        args.strategy = 'ddp'
        args.current_device = int(os.environ["SLURM_LOCALID"])
        print(f'Slurm found with {args.nodes} nodes, {args.devices} devices per node (current:{args.current_device}).')
    else:    
        args.nodes = 1
        # gpu
        if args.accelerator == 'gpu':
            args.devices = torch.cuda.device_count()
            if args.devices == 0:
                args.accelerator = 'cpu'
                print('No GPU found, switch to CPU mode.')
            elif args.devices == 1:
                args.current_device = 0
                print('Uising 1 GPU for training!')
            else:
                args.current_device = int(os.environ["LOCAL_RANK"])
                print(f'No slurm found! Uising {args.devices} local GPUs for training (current:{args.current_device})!')
        # cpu
        if args.accelerator == 'cpu':
            args.devices = 1
            args.current_device = 0
            print('Uising CPU for training!')

    ndevs = args.nodes * args.devices
    args.strategy = 'ddp' if ndevs>1 else 'auto'
    return args


def get_subset_datasets(dataset, subset_proportion=1):
    n_samples = int(len(dataset) * subset_proportion)
    rng = np.random.default_rng(seed=42)
    dataset = torch.utils.data.Subset(dataset, rng.choice(len(dataset), n_samples, replace=False))
    return dataset

# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)

    ###### 1 - parameter setting ######
    global args
    args = get_args_ft()
    if 'satmae' in args.pmn and args.n_channels==3:
        args.patch_size = PATCH_SIZE[args.pmn+'-3b']
        args.im_size = IMAGE_SIZE[args.pmn+'-3b']
    else:
        args.patch_size = PATCH_SIZE[args.pmn]
        args.im_size = IMAGE_SIZE[args.pmn]
    args.prt_tag = args.pmn
    pl.seed_everything(args.seed)

    ###### 2 - load data ######
    patch_size = datasets_settings_test[args.dataset_name]['input_size']
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
    if args.pmn in ['dofa']:
        args.n_channels = datasets_settings_test[args.dataset_name]['n_channels']
    n_channels = args.n_channels 

    # number of classes
    args.n_classes = len(class_names[args.dataset_name])

    if args.dataset_name == 'dw':
        train_dataset = DWOSMDataset(datasets_settings_test['dw']['data_path_ft'],
                                     mode=datasets_settings_test['dw']['mode'],
                                     n_bands=n_channels,
                                     transform=trans_tr)
        test_dataset = DWOSMDataset(datasets_settings_test['dw']['data_path'],
                                    mode=datasets_settings_test['dw']['mode'],
                                    n_bands=n_channels,
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
        
    # subset
    if args.dataset_proportion < 1:
        train_dataset = get_subset_datasets(train_dataset, args.dataset_proportion)
    sub_tag = f'sub{int(args.dataset_proportion*100)}' if args.dataset_proportion<1 else 'full'
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
    # add sub-tag to the group name
    if 'satmae' in args.pmn and args.n_channels==3:
        args.gname = f'{args.pmn}3B_{args.mt}_{args.frm}'
    else:
        args.gname = f'{args.pmn}_{args.mt}_{args.frm}'
    if args.pmn=='mfs2':
        args.gname += f'_{args.n_channels}B'
    if 'sub' in sub_tag: args.gname = f'{sub_tag}_{args.gname}_{args.eft_tag}'
    # lr
    ea = int(np.ceil(np.log10(1/args.lr)))
    eb = int(np.round(args.lr*10**ea))
    args.job_type = f'EP{args.epochs}_{args.dataset_name}_lr{eb}e{ea}' # f'{args.eft_tag}{args.eft_ep}_{feb}e{fea}_{args.dft_tag}_lr{eb}e{ea}'
    # optimizer
    if args.schdt!='none': 
        args.job_type += f'_{args.schdt}'
        if args.schdt=='step':
            for dv in args.schdv: args.job_type += f'_{int(dv)}'
        elif args.schdt=='exp':
            args.job_type += f'_{int(args.schdv[0]*100)}'
        elif args.schdt=='cos':
            args.job_type += f'_{int(args.schdv[0])}'
            if len(args.schdv)>1: 
                ea = int(np.ceil(np.log10(1/args.schdv[1])))
                eb = int(args.schdv[1]*10**ea)
                args.job_type += f'_f{eb}e{ea}'  # final_lr setting
    args.pdir = os.path.join(args.save_dir, args.gname, args.job_type)
    checkpoints_dir = os.path.join(args.pdir, 'checkpoints')
    # create directory
    p = pathlib.Path(args.pdir)
    p.mkdir(parents=True, exist_ok=True)

    # setting logger
    os.environ["WANDB_MODE"] = args.wmode
    wandb_logger = WandbLogger(project=args.pjn, entity=args.entity, 
                               group=args.gname, 
                               job_type=args.job_type,
                               name=f's{args.seed}_b{args.bs}',
                               save_dir=args.pdir) #??????


    ###### 3 - define the model ######
    torch.set_float32_matmul_precision('high')
    model = UNET_PL_FT(args)
    
    
    ###### 4 - train the model ######
    print('\nStart training...')
    callbacks = [ModelCheckpoint(dirpath = checkpoints_dir, every_n_epochs = args.save_inv, 
                                 # save_weights_only=True, 
                                 save_top_k=-1,  # Save all models
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
                         enable_progress_bar=True,
                         callbacks=callbacks)  

    trainer.fit(model, train_loader, test_loader)
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Training is finished|Total spent time:{t1-t0}!' )
    
    
    
if __name__ == '__main__':
    main()


