import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path


def get_args_1st_stage_train():
    parser = argparse.ArgumentParser(description='Train SAM2 with exact labels')
    # data settings
    parser.add_argument('--dataset_name', type=str, nargs='*', default=['dw', 'dynamicearthnet', 'openearthmap'], 
                        help='Names of the datasets used for training')
    parser.add_argument('--subset_proportion', type=float, default=1, help='Proportion of the dataset used for training')
    parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples used for training')

    # get common arguments
    parser = get_common_args(parser)

    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)

    # set saving dir
    args = set_log_files(args)

    return args

def get_args_train_w_unc():
    parser = argparse.ArgumentParser(description='Train SAM2 with exact labels')
    # data settings
    parser.add_argument('--dataset_name', type=str, nargs='*', default=['dw', 'dynamicearthnet', 'openearthmap'], 
                        help='Names of the datasets used for training')
    parser.add_argument('--subset_proportion', type=float, default=1, help='Proportion of the dataset used for training')
    parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples used for training')

    # uncertainty settings
    parser.add_argument('--use_no_weak_labels_as_logvar_input', dest='no_weak_input', action='store_true', default=False, help='Whether to use weak labels as logvar input or not')
    parser.add_argument('--weak_weight_rampup_end', type=int, default=100, help='Number of epochs for weak weight rampup')
    parser.add_argument('--weak_weight_rampup_start', type=int, default=0, help='Number of epochs for weak weight rampup')
    parser.add_argument('--weak_weight_final', type=float, default=1.0, help='Final weight of weak loss')
    parser.add_argument('--weak_uncertainty_lambda', dest='weak_unc_lambda', type=float, default=1.0, help='Weight of weak uncertainty scale loss')
    parser.add_argument('--weak_branch_with_detach_encoder', dest='weak_detach', action='store_true', default=False, help='Whether to detach encoder for weak branch or not')

    # rescale settings
    parser.add_argument('--rescale_weights', action='store_true', default=False, help='Whether to rescale weights or not')
    parser.add_argument('--rescale_weights_positive_mean', dest='weight_pos_mean', type=float, default=0.75, help='Positive mean for rescaling weights')
    parser.add_argument('--rescale_weights_negative_mean', dest='weight_neg_mean', type=float, default=0.25, help='Negative mean for rescaling weights')

    # get common arguments
    parser = get_common_args(parser)

    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)

    # set saving dir
    args = set_log_files(args, w_unc=True)

    return args


def get_args_1st_stage_test():
    parser = argparse.ArgumentParser(description='Test 1st stage models')
    # data settings
    parser.add_argument('--dataset_name', type=str, default='dw', help='Name of the test dataset')
    
    # get common arguments
    parser = get_common_args(parser)
    
    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)

    return args


def get_args_test_framework():
    parser = argparse.ArgumentParser(description='Test Framework without saving')
    # data settings
    parser.add_argument('--test_dataset_names', nargs='*', type=str, default=['dw', 'osm', 'multisenge', 'loveda', 'nyc', 'potsdam'], help='Names of the datasets used for testing')
    parser.add_argument('--test_batch_sizes', nargs='*', type=int, default=[30], help='Batch size for testing (len=1 or len(test_dataset_names))')

    # model setting
    parser.add_argument('--multiple_aux_decoder', type=int, default=1, help='Number of auxiliary decoders')
    parser.add_argument('--test_max_threshold', type=float, default=-1)
    parser.add_argument('--test_background_id', dest='test_bg_idx', type=int, default=0)
    
    # get common arguments
    parser = get_common_args(parser)

    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)

    return args


def get_args_fine_tune():
    parser = argparse.ArgumentParser(description='Fine-tuning')
    # data settings
    parser.add_argument('--dataset_name', type=str, default='dw', help='Names of the datasets used for fine-tuning')
    parser.add_argument('--dataset_proportion', type=float, default=1, help='Proportion of the dataset used for fine-tuning')
    parser.add_argument('--fine_tune_accuracy_branch', dest='fine_tune_acc', action='store_true', default=False, help='Whether to fine-tune accuracy branch or not')
    parser.add_argument('--fine_tune_checkpoint_path', dest='ft_wpath', type=str, default=None, help='Path of model weights to fine tune.')
    # for model weight optimization
    parser.add_argument('--freeze_backbones_only', action='store_true', default=False, help='To freeze two backbones only or the whole encoder (by default: to freeze the whole encoder).')
    # for binary loss settings
    parser.add_argument('--use_binary_loss', action='store_true', default=False, help='Whether to use binary loss or not')
    parser.add_argument('--use_class_multiplier', action='store_true', default=False, help='Whether to use class multiplier or not')

    # get common arguments
    parser = get_common_args(parser)

    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)
    
    return args


def get_args_ov():
    parser = argparse.ArgumentParser(description='Test Framework for OV')
    # data settings
    parser.add_argument('--test_dataset_names', nargs='*', type=str, default=['dw', 'osm', 'multisenge', 'loveda', 'nyc', 'potsdam'], help='Names of the datasets used for testing')
    parser.add_argument('--test_batch_sizes', nargs='*', type=int, default=[30], help='Batch size for testing (len=1 or len(test_dataset_names))')

    # model setting
    parser.add_argument('--multiple_aux_decoder', type=int, default=1, help='Number of auxiliary decoders')
    parser.add_argument('--test_max_threshold', type=float, default=-1)
    parser.add_argument('--test_background_id', dest='test_bg_idx', type=int, default=0)

    # proxy settings
    parser.add_argument('--clip_type', type=str, default='openai', help='Type of CLIP model')
    parser.add_argument('--clip_visual_type', type=str, default='ViT-B/16', help='Type of CLIP visual model')
    parser.add_argument('--clip_slide_stride', type=int, default=112, help='Stride of cropping')
    parser.add_argument('--clip_slide_crop', type=int, default=336, help='Size of crops')
    parser.add_argument('--vfm_model', type=str, default='weak', help='Type of VFM model for proxyclip')
    parser.add_argument('--weak_feature_type', dest='weak_feat_type', type=str, default='encoder_embed', choices=['encoder_embed', 'preout_embed'], help='Type of features of weak VFM model.')
    parser.add_argument('--max-confidence-threshold', dest='max_thr', type=float, default=0.6, help='Max confidence threshold for fusion.')
    # get common arguments
    parser = get_common_args(parser)

    # initialize arguments
    args = parser.parse_args()

    # computing environment parameters
    args = get_device_info(args)

    return args


def get_common_args(parser):
    # saving directory
    parser.add_argument('--save_dir', type=str, default='Results_test', help='Directory to save the results')  
    # data settings
    parser.add_argument('--patch_size', type=int, default=256, help='Random cropped Patch size before resizing to 1024.')
    parser.add_argument('--batch_size', dest='bs', type=int, default=5, help='Batch size')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=5, help='Batch size for test and validation sets')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, help='Number of workers for dataloaders')  
    # model settings
    parser.add_argument('--model-config-file', dest='config_file', type=str, default='sam2_hiera_l.yaml', help='Path to the model config file')
    parser.add_argument('--ckpt-path', type=str, default='checkpoints/sam2_hiera_large.pt', help='Path to the checkpoint file (sam2)')
    parser.add_argument('--ckpt-path-clip', type=str, default='checkpoints/RS5M_ViT-B-32.pt', help='Path to the checkpoint file (clip)') 
    parser.add_argument('--load-decoder-weights', dest='load_de', action='store_true', default=False, help='Whether to load decoder weights or not')
    parser.add_argument('--load-aux-decoder-weights', dest='load_aux_de', action='store_true', default=False, help='Whether to load auxiliary decoder weights or not (only used when load_decoder_weights is True)')
    parser.add_argument('--use-separate-auxiliary-decoder-for-weak-labels', dest='use_sep_aux_de_weak', action='store_true', default=False, help='Whether to use separate auxiliary decoder for each weakly labeled dataset or not')
    parser.add_argument('--use-auxiliary-decoder', dest='use_aux_decoder', action='store_true', default=False, help='Whether to use auxiliary decoder (double decoders) or not')
    parser.add_argument('--share-short-connect-weights', dest='share_connect', action='store_true', default=False, help='Whether to share short connection weights or not when using auxiliary decoder')
    parser.add_argument('--expand-decoder-dimension', dest='de_expand_dim', action='store_true', default=False, help='Whether to expand decoder dimension or not (False: //4, //8; True: *1, *2 to align with CLIP-base dimension)')
    parser.add_argument('--use-cross-attention-two-way-trans', dest='opt_two', action='store_true', default=False, help='Whether to use noise mask embedding as an additional input in the two-way transformer (option 2) or not (option 1).')
    parser.add_argument('--use-feature-norm', action='store_true', default=False, help='Whether to use feature normalization or not when generating masks')
    # alignment to clip
    parser.add_argument('--align-to-clip', action='store_true', default=False, help='Whether using distillation losses to align the model to clip feature space or not')
    parser.add_argument('--clip-model-type', type=str, choices=['vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP'], default='ClearCLIP', help='Type of clip model used for feature alignment.')
    parser.add_argument('--align-loss-type', type=str, choices=['mse', 'l1', 'kl', 'ce', 'cos'], default='mse', help='Type of loss used for feature alignment.')
    # logvar smoother settings
    parser.add_argument('--use_logvar_smooth', action='store_true', default=False, help='Whether to use logvar smoother or not')
    parser.add_argument('--fix_logvar_smooth', action='store_true', default=False, help='Whether to fix logvar smoother or not if using logvar smoother')
    # parser.add_argument('--number_of_multiple_outputs', dest='n_mout', type=int, default=3, help='Number of multiple outputs')
    parser.add_argument('--use-noise-mask-loss', action='store_true', default=False, help='Whether to use the mask and eixstance mask on the branch where noise mask is used as input')
    parser.add_argument('--mute-adaptor', action='store_true', default=False, help='Whether to use adaptor or not')
    parser.add_argument('--mute-high-frequency', dest='mute_highfreq', action='store_true', default=False, help='Whether to mute the high frequency component or not in the adapter')
    parser.add_argument('--dofa-type', type=str, default=None, choices=['vit_base_patch16', 'vit_large_patch16', 'None'], help='Type of DoFA model to use. Default is None (not using dofa).')
    parser.add_argument('--dofa-ckpt-path', type=str, default=None, help='Path to the checkpoint file (dofa)')
    parser.add_argument('--dofa-wo-rgb', action='store_true', default=False, help='Whether to use dofa to extract features from rgb or not')
    parser.add_argument('--dofa-unfix-weights', action='store_true', default=False, help='Whether to unfix dofa weights or not')
    parser.add_argument('--dofa-lr-scale', type=float, default=1.0, help='Learning rate scale for DOFA if dofa-unfix-weights is True')
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
    parser.add_argument('--cross_entropy_loss_weight', dest='wce', type=float, default=1.0, help='Weight of cross entropy loss')
    parser.add_argument('--use_class_weights', action='store_true', default=False, help='Use class weights or not to weight binary losses.')
    parser.add_argument('--class_weights_scale', type=float, default=10, help='Scale of class weights')
    parser.add_argument('--iou_no_object_weight', dest='wiou_no', type=float, default=0.5, help='Weight of iou loss for no object')
    parser.add_argument('--use_multiclass_loss', action='store_true', default=False, help='Use multiclass loss or not')
    parser.add_argument('--use_binary_class_scale', action='store_true', default=False, help='Use number of classes as the binary loss scale or not.')
    parser.add_argument('--amp', action='store_false', default=True, help='Use mixed precision or not')
    parser.add_argument('--scheduler_type', dest='schdt', type=str, default='none', choices=['none','cos','step','exp'], help='Type of scheduler to use')
    parser.add_argument('--scheduler_values', dest='schdv', type=float, nargs='*', default=[], help='Values of scheduler to use (the epoch indexes for step scheduler; number of iterations for cosine scheduler)')
    # dropout
    parser.add_argument('--use_mc_dropout', action='store_true', default=False, help='Whether to use dropout for training')
    parser.add_argument('--p_mc_dropout', type=float, default=0.5, help='Dropout rate for training')
    # wandb settings
    parser.add_argument('--project_name', dest='pjn', type=str, default='NLPR_SSL4EO', help='Name of the wandb project to record statistics.')
    parser.add_argument('--entity_name', dest='entity', type=str, default='vikki023', help='Name of the entity of wandb.')
    parser.add_argument('--wandb_mode', dest='wmode', type=str, default='online', choices=["online", "offline", "disabled"], help='Setting of wandb.init modes ("online", "offline" or "disabled")')
    # others
    parser.add_argument('--display_interval', dest='display_inv', type=int, default=1, help='Set the display interval during the training')
    parser.add_argument('--save_interval', dest='save_inv', type=int, default=10, help='Set the save interval during the training')
    parser.add_argument('--seed', type=int, default=42)
    # for test
    parser.add_argument('--test_only', action='store_true', default=False, help='Only test models.')
    parser.add_argument('--test_weight_path', dest='ts_wpath', type=str, default=None, help='Path of test model weights.')
    parser.add_argument('--test_use_multiple_prompts', action='store_true', default=False, help='Whether to use multiple prompts or not')
    # for debugging tag
    parser.add_argument('--debug', type=str, default='', help='Debugging tag')
    return parser


def set_log_files(args, w_unc=False):
    expand_dim_tag = 'E' if args.de_expand_dim else ''
    # group name
    noise_mask_loss_tag = 'l' if args.use_noise_mask_loss else ''
    if args.use_sep_aux_de_weak:
        aux = f'_sepaux_attcp'
    else:
        ctag = 'c' if args.share_connect else ''
        aux = f'_aux{noise_mask_loss_tag}{ctag}_attcp' if args.use_aux_decoder else '_attcp'
    # adaptor
    args.use_adaptor = not args.mute_adaptor
    if args.use_adaptor:
        args.gname = f'FT{expand_dim_tag}'
        # high-frequency option
        if args.mute_highfreq:
            args.gname += '_noHF'
        # dofa option
        if args.dofa_type in [None, 'None']:
            args.gname += aux
        else:
            if args.dofa_unfix_weights:
                dofa_active = 'A'
                if args.dofa_lr_scale!=1.0:
                    dofa_active += f'x{args.dofa_lr_scale}'  
            else:
                dofa_active = ''    
            if args.dofa_wo_rgb:
                dofa_active += 'woRGB'               
            args.gname += f'_dofa{args.dofa_type[4]}{dofa_active}{aux}'
    else:
        args.gname = f'FT{expand_dim_tag}_woAd{aux}'

    # training size
    subset = f'_sub{args.subset_proportion}' if args.subset_proportion<1 else ''
    args.gname += f'{subset}_max{args.max_dataset_size}' if args.max_dataset_size is not None else ''

    # using pretrained decoder or not
    if args.load_de:
        args.gname += '_deIaux' if args.load_aux_de else '_deI'   
    else:
        args.gname += '_deN'
    # using multiclass loss or not
    args.gname += '_mcloss' if args.use_multiclass_loss else ''
    # using binary loss scale or not
    args.gname += '_bscale' if args.use_binary_class_scale else ''
    # language model type
    args.gname += f'_{args.ckpt_path_clip.split("/")[-1][:4]}'
    # # # # # # For debugging stage # # # # # 
    # args.gname += 'new' 

    # alignment settings
    if args.align_to_clip:
        args.gname += f'_a{args.clip_model_type}_{args.align_loss_type}'
        
    # job type
    args.job_type = f'{int(args.wfocal)}F_{int(args.wdice)}D_{int(args.wce)}C' 
    if args.use_class_weights: args.job_type += f'_clsw{int(args.class_weights_scale)}'
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
            args.job_type += f'_{int(args.schdv[0]*1000)}'
        elif args.schdt=='cos':
            if args.schdv[0]>0:
                args.job_type += f'_w{int(args.schdv[0])}_cos{int(args.schdv[1])}'
            else:
                args.job_type += f'_cos{int(args.schdv[1])}'
            if len(args.schdv)>2: 
                ea = int(np.ceil(np.log10(1/args.schdv[2])))
                eb = int(args.schdv[2]*10**ea)
                args.job_type += f'_f{eb}e{ea}'  # final_lr setting
    # feature normalization setting
    if args.use_feature_norm: 
        args.job_type += '_norm'
    # dropout settings
    if args.use_mc_dropout:
        args.job_type += f'_drop{int(100*args.p_mc_dropout)}'
    # for uncertainty part
    if w_unc:
        args.gname += f'_unc{args.weak_weight_final}'
        args.job_type += f'_rmup{args.weak_weight_rampup_start}_{args.weak_weight_rampup_end}_l{args.weak_unc_lambda}'
        if args.weak_detach: args.job_type += '_denc'
        args.gname += '_wowl' if args.no_weak_input else ''
        if args.rescale_weights:
            args.gname += f'_rw_p{args.weight_pos_mean}_n{args.weight_neg_mean}'
        if args.use_logvar_smooth:
            args.job_type += '_fsmt' if args.fix_logvar_smooth else '_asmt'
    # # for debugging tag
    # args.job_type += args.debug

    # saving dirs
    if args.opt_two and 'opt2' not in args.save_dir:
        args.save_dir += '_opt2'
    # for test only
    # if args.test_only: 
    #     # parent dir
    #     if args.ts_wpath is None:
    #         args.pdir = os.path.join(args.save_dir, args.gname) 
    #         ep = 0 
    #     else:
    #         args.pdir = os.path.join(args.save_dir, args.gname, args.job_type, f"mcr_{args.mcr}_seed_{args.seed}_bs_{args.bs}")
    #         ep_str = args.ts_wpath.split('epoch=')[1].split('-')[0]
    #         ep = int(ep_str)
    #     # test dir
    #     args.pdir = os.path.join(args.pdir, f'test_ep{ep}')
    #     Path(args.pdir).mkdir(parents=True, exist_ok=True)
        
    #     # log outputs to txt file
    #     acc_txt = os.path.join(args.pdir, 'outputs.txt')
    #     sys.stdout = open(acc_txt, 'w')
    # for fine-tuning training
    # else:
    if not args.test_only:
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
        args.devices = int(os.environ["SLURM_GPUS_ON_NODE"]) # [0]) #.split('(')[0])
        args.current_device = int(os.environ["SLURM_LOCALID"])
        print(f'Slurm found with {args.nodes} nodes, {args.devices} devices per node (current:{args.current_device}).')
        args.ntasks = args.nodes * args.devices
        # args.ntasks = int(os.environ["SLURM_NTASKS"])
        # print(f'Number of tasks: {args.ntasks}')
        # assert args.ntasks == args.nodes * args.devices, 'Number of tasks is not equal to the number of devices!'
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
                print('Using 1 GPU for training!')
            else:
                args.current_device = int(os.environ["LOCAL_RANK"])
                print(f'No slurm found! Uising {args.devices} local GPUs for training (current:{args.current_device})!')
        # cpu
        if args.accelerator == 'cpu':
            args.devices = 1
            args.current_device = 0
            print('Using CPU for training!')
            # args.devices = os.cpu_count()
            # print(f'Uising {args.devices} CPU cores for training!')
        args.ntasks = args.devices

    if args.ntasks>1: 
        args.strategy = 'ddp_find_unused_parameters_true' # if args.use_aux_decoder else 'ddp' 
    else:
        args.strategy = 'auto'
    return args