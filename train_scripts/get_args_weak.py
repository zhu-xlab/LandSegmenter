import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path


def get_args_ssl4eodw():
    parser = argparse.ArgumentParser(description='Fine-tune SAM2 with SSL4EO data')
    # data settings
    parser.add_argument('--data_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_validation.lmdb')
    parser.add_argument('--data_name_s1', dest='dns1', type=str, default=None)
    parser.add_argument('--data_name_s2', dest='dns2', type=str, default=None)
    parser.add_argument('--data_name_weak_label', dest='dnwl', type=str, default=None)
    parser.add_argument('--test_path', type=str, default='E:/Datasets/DFC2020/lmdb/DFC2020_s12_test.lmdb')

    # coteaching settings
    parser.add_argument('--rampup_type', type=str, default='none', choices=['none', 'linear', 'sigmoid'], help='Type of rampup')
    parser.add_argument('--rampup_length', type=int, default=0, help='Length of rampup')
    parser.add_argument('--init_forget_rate', type=float, default=0, help='Initial forget rate')
    parser.add_argument('--forget_rate', type=float, default=0, help='Final forget rate')

    # get common arguments
    parser = get_common_args(parser)

    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)

    # set saving dir
    args = set_log_files(args)

    return args    


def get_common_args(parser):
    # saving directory
    parser.add_argument('--save_dir', type=str, default='Results_test', help='Directory to save the results')  
    # data settings
    parser.add_argument('--input_type', dest='it', type=str, default='s2', choices=['s1', 's2', 's12'], help='Type of input data')
    parser.add_argument('--n_channels', type=int, default=3, help='Number of input channels.') 
    parser.add_argument('--patch_size', type=int, default=256, help='Random cropped Patch size before resizing to 1024.')
    parser.add_argument('--n-prompts', type=int, default=[9], nargs='*', help='Number of prompts')
    parser.add_argument('--validation', dest='val', type=float, default=0., help='Percent of the data that is used as validation (0-1.0)')
    parser.add_argument('--batch_size', dest='bs', type=int, default=5, help='Batch size')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=5, help='Batch size for test and validation sets')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, help='Number of workers for dataloaders')  
    parser.add_argument('--n_train', dest='ntr', type=int, default=-1, 
                        help='Number of training samples used (1-total number of trainig samples, default=-1 to use all of samples for training)')
    # model settings
    parser.add_argument('--model-config-file', dest='config_file', type=str, default='sam2_hiera_l.yaml', help='Path to the model config file')
    parser.add_argument('--ckpt-path', type=str, default='checkpoints/sam2_hiera_large.pt', help='Path to the checkpoint file (sam2)')
    parser.add_argument('--ckpt-path-clip', type=str, default='checkpoints/RemoteCLIP-ViT-B-32.pt', help='Path to the checkpoint file (clip)') 
    parser.add_argument('--load-decoder-weights', dest='load_de', action='store_true', default=False, help='Whether to load decoder weights or not')
    parser.add_argument('--resume', type=str, default=False, help='Path of the checkpoint file from which the training is resumed.')
    # device settings
    parser.add_argument('--accelerator', type=str, default='cpu', choices=['cpu','gpu'], help='Type of used accelerator')
    parser.add_argument('--slurm', action='store_true', default=False, help='Whether to use slurm or not')
    # general training settings
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--optimizer', dest='opt', type=str, default='adam', choices=['adam','sgd', 'adamw'], help='Type of used optimizer.')
    parser.add_argument('--learning_rate', '-l', dest='lr', metavar='LR', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--focal_loss_weight', dest='wfocal', type=float, default=10.0, help='Weight of focal loss')
    parser.add_argument('--dice_loss_weight', dest='wdice', type=float, default=1.0, help='Weight of dice loss')
    parser.add_argument('--amp', action='store_false', help='Use mixed precision or not')
    parser.add_argument('--scheduler_type', dest='schdt', type=str, default='none', choices=['none','cos','step','exp'], help='Type of scheduler to use')
    parser.add_argument('--scheduler_values', dest='schdv', type=float, nargs='*', default=[], help='Values of scheduler to use (the epoch indexes for step scheduler; number of iterations for cosine scheduler)')
    # wandb settings
    parser.add_argument('--project_name', dest='pjn', type=str, default='NLPR_SSL4EO', help='Name of the wandb project to record statistics.')
    parser.add_argument('--entity_name', dest='entity', type=str, default='vikki023', help='Name of the entity of wandb.')
    parser.add_argument('--wandb_mode', dest='wmode', type=str, default='online', choices=["online", "offline", "disabled"], help='Setting of wandb.init modes ("online", "offline" or "disabled")')
    # others
    parser.add_argument('--display_interval', dest='display_inv', type=int, default=1, help='Set the display interval during the training')
    parser.add_argument('--save_interval', dest='save_inv', type=int, default=10, help='Set the save interval during the training')
    parser.add_argument('--seed', type=int, default=42)
    # for test
    parser.add_argument('--only_test', action='store_true', default=False, help='Only test models.')
    parser.add_argument('--test_weight_path', dest='ts_wpath', type=str, default=None, help='Path of test model weights.')
    return parser


def set_log_files(args):
    # group name
    if args.ntr>0: 
        args.gname = f'test_coteach_DW_{args.ntr}'
    else:
        args.gname = f'test_coteach_DW_all'
    # coteach settings
    args.gname += f'_rampup_{args.rampup_type}_{args.rampup_length}_fr_{args.init_forget_rate}_to_{args.forget_rate}'
    # # using pretrained decoder or not
    # args.gname += '_deI' if args.load_de else '_deN'
        
    # job type
    args.job_type = f'{int(args.wfocal)}F_{int(args.wdice)}D' 
    # lr
    ea = int(np.ceil(np.log10(1/args.lr)))
    eb = int(np.round(args.lr*10**ea))
    args.job_type += f'_lr{eb}e{ea}' # f'{args.eft_tag}{args.eft_ep}_{feb}e{fea}_{args.dft_tag}_lr{eb}e{ea}'
    # optimizer
    if args.schdt!='none': 
        args.job_type += f'_{args.schdt}'
        if args.schdt=='step':
            for dv in args.schdv: args.job_type += f'_{int(dv)}'
        elif args.schdt=='exp':
            args.job_type += f'_{int(args.schdv[0]*100)}'
        elif args.schdt=='cos':
            if args.schdv[0]>0:
                args.job_type += f'_w{int(args.schdv[0])}_cos{int(args.schdv[1])}'
            else:
                args.job_type += f'_cos{int(args.schdv[1])}'
            if len(args.schdv)>2: 
                ea = int(np.ceil(np.log10(1/args.schdv[2])))
                eb = int(args.schdv[2]*10**ea)
                args.job_type += f'_f{eb}e{ea}'  # final_lr setting
    
    # saving dirs
    # for test only
    if args.only_test: 
        # parent dir
        if args.ts_wpath is None:
            args.pdir = os.path.join(args.save_dir, args.gname) 
            ep = 0 
        else:
            args.pdir = os.path.join(args.save_dir, args.gname, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}_bs_{args.bs}")
            ep_str = args.ts_wpath.split('epoch=')[1].split('-')[0]
            ep = int(ep_str)
        # test dir
        args.pdir = os.path.join(args.pdir, f'test_ep{ep}')
        Path(args.pdir).mkdir(parents=True, exist_ok=True)
        
        # log outputs to txt file
        acc_txt = os.path.join(args.pdir, 'outputs.txt')
        sys.stdout = open(acc_txt, 'w')
    # for fine-tuning training
    else:
        # parent dir
        args.pdir = os.path.join(args.save_dir, args.gname, args.job_type, f"seed_{args.seed}_bs_{args.bs}")
        # checkpoint saving dir
        args.checkpoints_dir = os.path.join(args.pdir,"checkpoints")
        Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    
        # statistic recording in txt files
        # record params into txt file
        with open(f'{args.pdir}/params.txt', 'w') as ft:
            json.dump(args.__dict__, ft)
    return args

 
def get_device_info(args):
    if args.slurm:
        args.slurm = "SLURM_JOB_ID" in os.environ
        
    if args.slurm:
        args.nodes = int(os.environ["SLURM_NNODES"])
        args.devices = int(os.environ["SLURM_TASKS_PER_NODE"])
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
            # args.devices = os.cpu_count()
            # print(f'Uising {args.devices} CPU cores for training!')

    ndevs = args.nodes * args.devices
    args.strategy = 'ddp' if ndevs>1 else 'auto'
    return args