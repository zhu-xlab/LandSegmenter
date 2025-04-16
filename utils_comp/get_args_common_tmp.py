
def get_common_args(parser):
    # for reproducibility
    parser.add_argument('--monte_carlo_run', type=int, dest='mcr', default=1, help='Number of Monte Carlo runs')
    # saving directory
    parser.add_argument('--save_dir', type=str, default='Results_test')  
    # model settings
    parser.add_argument('--model_framework', dest='frm', type=str, default='unet', choices=['unet','deeplabv3','deeplabv3plus','pspnet','linknet','fpn'],
                        help='Type of used model framework.')
    parser.add_argument('--model_type', dest='mt', type=str, default='resnet50',
                        help='Type of used models. It is mainly utilized to define the encoder part') 
    parser.add_argument('--n_channels', type=int, default=13,
                        help='Number of input channels.') 
    parser.add_argument('--n_classes', type=int, default=8,
                        help='Number of predefined classese.')
    parser.add_argument('--loss_type', type=str, default='cd',
                        help='Type of used segmentation loss function for training. ' +
                        'Specifically, \'c\' represents CrossEntropy loss, \'d\' represents Dice loss, \'f\' represents focal loss.') 
    parser.add_argument('--resume', type=str, default=False,
                        help='Path of the checkpoint file from which the training is resumed.')
    # general data settings 
    parser.add_argument('--if_mask',  action='store_true',
                        help='Indicate whether using mask strategy to train model') 
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.10, 
                        help='Percent of the data that is used as validation (0-1.0)')
    parser.add_argument('--batch_size', dest='bs', type=int, default=5, 
                        help='Batch size')
    parser.add_argument('--test_batch_size', dest='tbs', type=int, default=5, 
                        help='Batch size for test and validation sets')
    parser.add_argument('--num_workers', '-n', metavar='NW', type=int, default=0, 
                        help='Number of workers for dataloaders')   
    # label smoothing
    parser.add_argument('--label_smoothing', dest='ls', action='store_true',
                        help='Whether to use label smoothing or not')
    parser.add_argument('--label_smoothing_factor', dest='lsf', type=float, nargs='*', default=[],
                        help='Factors of label smoothing: the first one is for temporal-spatial prior, the second (if set) is for the fixed item.')
    parser.add_argument('--label_smoothing_exclude_class', dest='ls_exclude', type=int, nargs='*', default=[],
                        help='Classes to be excluded when using label smoothing')
    parser.add_argument('--label_smoothing_prior_type', dest='ls_prior_type', type=str, default='ts', choices=['ts','t','s'],
                        help='Type of prior used for label smoothing')
    # device settings
    parser.add_argument('--accelerator', type=str, default='cpu', choices=['cpu','gpu'], 
                        help='Type of used accelerator')
    parser.add_argument('--slurm', action='store_true', default=False, 
                        help='Whether to use slurm or not')
    # general training settings
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--optimizer', dest='opt', type=str, default='adam', choices=['adam','sgd'],
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
    # parser.add_argument("--print_to_log", action="store_false", help="If true, directs std-out to log file")
    parser.add_argument("--batch_to_wandb", action="store_true", help="If true, log batch-wise training results to wandb")
    parser.add_argument('--seed', type=int, default=42)
    
    return parser


def get_common_finetune_args(parser):
    parser.add_argument('--dataset', type=str, default='DFC', choices=['DFC','DW','OSCD', 'OSM'], help='Dataset name')
    # training size
    parser.add_argument('--n_train', dest='ntr', type=int, default=0, 
                        help='Number of training samples used (1-total number of trainig samples, default=-1 to use all of samples for training)')
    # pretrain settings
    parser.add_argument('--pretrain_tag', dest='prt_tag', type=str, default='nss2', 
                        choices=['rand','imagenet','ssl','swsl','ns2cls','ns3cls','nss1','nss2', 'dinot','dinos','mocoq','mocok','mfs1','mfs2','lfs1','lfs2','satlas'],  
                        help='Tag of pretraining type for recording.') 
    parser.add_argument('--weight_path', dest='wpath', type=str, default=r'ssl_models\new\ns\ssl4eo\epoch=149-step=15000.ckpt', help='Path of pretrained model weights.')
    parser.add_argument('--weight_epoch', dest='wep', type=int, default=0, help='Epoch index of pretrained model weights.')
    parser.add_argument('--weight_batch_size', dest='wbs', type=int, default=128, help='Batch size used in pretraining.')
    # label smoothing
    parser.add_argument('--weight_label_smooth', dest='wls', action='store_true', help='Whether using label smoothing in pretraining.')
    parser.add_argument('--weight_label_smoothing_factor', dest='wlsf', type=float, nargs='*', default=[],
                        help='Factors of label smoothing in pretraining: the first one is for temporal-spatial prior, the second (if set) is for the fixed item.')
    parser.add_argument('--weight_label_smoothing_exclude_class', dest='wls_exclude', type=int, nargs='*', default=[],
                        help='Classes to be excluded when using label smoothing in pretraining')
    parser.add_argument('--weight_label_smoothing_prior_type', dest='wls_prior_type', type=str, default='ts', choices=['ts','t','s'],
                        help='Type of prior used for label smoothing in pretraining')
    # for multi-modal pretraining
    # sample selection
    parser.add_argument('--weight_sample_selection', dest='wss', action='store_true', default=False,
                        help='Whether to use sample selection')
    parser.add_argument('--weight_sample_selection_rmup_func', dest='wss_rm_func', type=str, choices=['linear', 'exp', 'none'], default='none',
                        help='Function used for sample selection proportion ramp-down')
    parser.add_argument('--weight_sample_selection_rmdown_epoch', dest='wss_nep', type=int, default=50,
                        help='Number of epochs when sample selection proportion ramps-down')
    parser.add_argument('--weight_sample_selection_prop', dest='wss_prop', type=float, default=0.5,
                        help='Proportion of samples to be selected finally')
    parser.add_argument('--weight_sample_selection_confidence_type', dest='wss_ctype', type=str, choices=['ce', 'gini'], default='ce',
                        help='Type of confidence used for sample selection')
    # finetune settings
    parser.add_argument('--weight_load_tag', dest='wload_tag', type=str, default='encoder', choices=['encoder','wocls','all','none'],
                        help='Tag of weight loading type for recording.')
    parser.add_argument('--encoder_finetune_tag', dest='eft_tag', type=str, default='etr', choices=['efix','eft','etr'],
                        help='Tag of finetuning type of encoder for recording.')
    parser.add_argument('--decoder_finetune_tag', dest='dft_tag', type=str, default='dsam', choices=['dsam','dasc','ddes'],
                        help='Tag of finetuning type of decoder for recording.')
    parser.add_argument('--lr_varying_factor', dest='lr_vf', type=float, default=0.8,
                        help='Varying gactor of learning rate when using different lrs for different layers')
    parser.add_argument('--scheduler_type', dest='schdt', type=str, default='none', choices=['none','cos','step','exp'], 
                        help='Type of scheduler to use')
    parser.add_argument('--scheduler_values', dest='schdv', type=float, nargs='*', default=[],
                        help='Values of scheduler to use (the epoch indexes for step scheduler; number of iterations for cosine scheduler)')
    parser.add_argument('--encoder_finetune_epoch', dest='eft_ep', type=int, default=0, 
                        help='Epoch index when starting to finetune the encoder if eft_tag is \'eft\'')
    # parser.add_argument('--encoder_finetune_lr', dest='eft_lr', type=float, default=0.001,
    #                     help='Learning rate for encoder (the whole net) finetuning')
    # for test
    parser.add_argument('--only_test', action='store_true', default=False,
                        help='Only test models.')
    parser.add_argument('--test_weight_path', dest='ts_wpath', type=str, default=None, 
                        help='Path of test model weights.')
    
    return parser


def get_common_pretrain_args(parser):
    
    parser.add_argument('--pre-train', dest='prt', action='store_true', default=False, 
                        help='Whether to load pretrained model weights for initialization or resume training if resume is not False.')
    # data settings 
    parser.add_argument('--experiment', type=str, choices=['gt_labels','ns_labels'], default='ns_labels', 
                        help='Type of labels used for training')
    parser.add_argument('--gt_available', dest='gt_avail', action='store_true', 
                        help='Whether having access to ground truth labels or not when training with noisy labels.')
    # scheduler settings
    parser.add_argument('--lr_adjust', dest='schd', type=float, default=10, 
                        help='The patience used to adjust lr according to validation loss. No scheduler is used if schd<=0.')
    return parser