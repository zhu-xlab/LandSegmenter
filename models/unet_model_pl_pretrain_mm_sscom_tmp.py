import os
import sys
import wandb
import numpy as np
# appending self-defined package path
sys.path.append('../Codes')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# import segmentation models pytorch package
from segmentation_models_pytorch.losses import FocalLoss

# import lightening
import pytorch_lightning as pl

# self-defined modules
import utils.evaluates as evl
from models.multimodel.unet_midFusion import Unet_midF_2branch, Unet_midF_3branch
from models.multimodel.unet_lateFusion import Unet_lateF_2branch, Unet_lateF_3branch
from utils.utils import sigmoid_rampup, linear_rampup
from utils.myLosses import SoftDiceLoss


KEY_IMG = 'img'
KEY_GT = 'gt'
KEY_NS = 'ns'
KEY_METRICS = ['iou', 'f1', 'precise', 'recall', 'oa']
DEBUG = False

# # # # for model loading # # # # 
def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd

def send_optim_tensor_gpu(optimizer,device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == 'step': continue
                state[k] = v.to(device=device)
    return
# # # # for model loading # # # # 
    
    
class Unet_mm(pl.LightningModule):
    # 1. training setup
    def __init__(self, args):
        """
        Initializes a UNet model with the specified architecture and loss functions.

        Args:
            args: An object containing the following attributes:
                - mt: A string representing the name of the encoder to use (e.g. mobilenet_v2 or efficientnet-b7).
                - resume: A string representing the path to a checkpoint file to resume training from, or one of the following strings: 'imagenet', 'ssl', 'swsl' to use pre-trained weights for encoder initialization.
                - prt: A boolean indicating whether the resumed model is a pre-trained one for initialization or for continued training.
                - n_channels: An integer representing the number of input channels (e.g. 1 for gray-scale images, 3 for RGB, etc.).
                - n_classes: An integer representing the number of output channels (i.e. the number of classes in the dataset).
                - loss_type: A string representing the type of loss function(s) to use, where 'c' stands for Cross Entropy, 'd' stands for Dice, and 'f' stands for Focal loss.
                - if_mask: A boolean indicating whether the dataset contains masks.
                - experiment: A string representing the type of experiment label to use, where 'ns_labels' stands for non-seismic labels and 'gt_labels' stands for ground truth labels.
                - gt_avail: A boolean indicating whether ground truth labels are available when training with noisy labels.
                - cal_tr: A boolean indicating whether to calculate training metrics.
                - opt: A string representing the optimizer to use (e.g. Adam, SGD, etc.).
                - lr: A float representing the learning rate.
                - schd: A string representing the learning rate scheduler to use (e.g. StepLR, CosineAnnealingLR, etc.).
                - batch_to_wandb: An integer representing the number of batches to log to Weights & Biases.
        """
        super().__init__()
        # pl.seed_everything(args.seed)
        self.last_epoch = 0
        self.batch_wise = False #False  # whether calculate metrics batch-wisely or as a whole
        # # # # model architecture # # # #
        self.n_classes = args.n_classes
        self.n_channels_1 = 2 # for s1
        self.n_channels_2 = args.n_channels
        self.use_com_feat = args.use_com_feat
        self._get_model(args)

        # # # # segmentation loss # # # #
        self.cl_type = args.cl_type
        self.cl_weight = args.cl_weight
        self._get_loss(args.n_classes, args.loss_type, args.if_mask)
        
        # # # # label smoothing settings # # # #
        self.ls = args.ls
        self.lsf = args.lsf
        
        # # # # sample selection settings # # # #
        self.ss = args.ss
        self.ss_ctype = args.ss_ctype
        self.ss_nep = args.ss_nep  
        self.ss_prop = args.ss_prop
        self.ss_rmup_func = self._get_rampup_func(args.ss_rm_func)
        
        # # # # other hyperparameters # # # # 
        self.opt = args.opt
        self.lr = args.lr
        # self.schdt = args.schdt
        self.schd = args.schd
        self.if_mask = args.if_mask
        self.log_batch = args.batch_to_wandb
        assert args.experiment in ['ns_labels','gt_labels'], 'Experiment label type is wrong!'
        self.experiment = args.experiment
        self.gt_avail = args.gt_avail
        self.cal_tr = args.cal_tr
        self.current_device = args.current_device
        self.display_inv = args.display_inv
        self.cls_dict = args.cls_dict
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        
        # for recording confidence values at each batch in on_train_batch_end
        # self.training_step_confs = [torch.zeros((2,self.n_classes), dtype=float) for _ in range(2)]
        # for recording tp, fp, fn, tn in on_validation_epoch_end
        if not self.batch_wise: 
            self.validation_step_outputs = {}
            for k in ['s1', 's2']:
                self.validation_step_outputs[k] = {'tp': [], 'fp': [], 'fn': [], 'tn': []}
    
     
    # 2. forward propagation
    def forward(self, x1, x2):
        outs = self.net(x1, x2)
        return outs
    
    
    # 3. Optimizer
    def configure_optimizers(self):
        # optimizer
        if self.opt=='adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
            print('Adam is used as optimizer!')
        elif self.opt=='sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            print('SGD is used as optimizer!')
        else:
            raise ValueError("Please provide correct optimizer type (adam or sgd)!")
        # resume optimizer state
        if self.opt_w is not None:
            try:
                optimizer.load_state_dict(self.opt_w)
            except:
                raise TypeError("The type of optimizer stored in ckpt file is not compitable to the used optimizer!")
            # # send optimizer tensor to gpu
            # if args.opt=='adam': send_optim_tensor_gpu(optimizer,device)
            print(f"Training is resumed at epoch {self.last_epoch} from '{self.last_epoch}'!")
        # scheduler
        if self.schd > 0:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.schd, factor=0.5, threshold=0.0001)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "v_loss"}
        else:
            return optimizer
    
    
    # 4. training step
    def training_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        if self.ls: smooth_prior = batch['smooth_prior']
        x1, x2 = x[:,:self.n_channels_1], x[:,self.n_channels_1:]
        del x
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x1, x2)
        if self.if_mask: 
            discard_masks = batch['mask_discard'].squeeze(axis=1)
            y[discard_masks!=0] = self.n_classes
        # sample selection mask (co-teaching)
        if self.ss:
            # prediction confidence: ce or gini
            pred_confs = [self._get_single_pred_confidence_mask(pred_logits[i]) for i in range(2)]
            # label confidence
            lbl_confs = [self._get_single_label_confidence_mask(pred_logits[i], y) for i in range(2)]
            # common information enhanced confidence
            pred_confs = self._get_common_enhanced_confidence_masks(pred_confs)
            lbl_confs = self._get_common_enhanced_confidence_masks(lbl_confs)
            
            # sample selection masks
            # for label confidence (segmentation part): weights for the first proportion (ss_ratio) are set to 1, 
            # while the weights for the rest uncertainty ones are set to (0-1) normalized by the maximum confidence in this part
            ss_masks_lbl = [self._get_selection_mask_by_label_confidence(lbl_confs[i], y, class_balanced=True) for i in range(2)]
            # for prediction confidence (consistency part)
            ss_masks_pred = [self._get_selection_mask_by_pred_confidence(pred_confs[i]) for i in range(2)]
            # # # # # for debugging # # # # #
            if DEBUG and self.current_epoch<5:
                print(f"EP{self.current_epoch}|ss_masks_lbl_0: {(ss_masks_lbl[0]==1).sum()} samples are selected with weight=1! min:{ss_masks_lbl[0].min()}, max:{ss_masks_lbl[0].max()}")
                print(f"EP{self.current_epoch}|ss_masks_lbl_1: {(ss_masks_lbl[1]==1).sum()} samples are selected with weight=0! min:{ss_masks_lbl[1].min()}, max:{ss_masks_lbl[1].max()}")
                print(f"EP{self.current_epoch}|ss_masks_pred_0: {(ss_masks_pred[0]>0).sum()} samples are selected with weight>0! min:{ss_masks_pred[0].min()}, max:{ss_masks_pred[0].max()}")
                print(f"EP{self.current_epoch}|ss_masks_pred_1: {(ss_masks_pred[1]>0).sum()} samples are selected with weight>0! min:{ss_masks_pred[1].min()}, max:{ss_masks_pred[1].max()}")
                # print(f"EP{self.current_epoch}|{ss_masks_lbl[0].shape}, {ss_masks_lbl[0].requires_grad}, {ss_masks_lbl[1].requires_grad}, {ss_masks_pred[0].requires_grad}, {ss_masks_pred[1].requires_grad}")
            # # # # # for debugging # # # # #
            # calculate loss
            loss = self._calculate_loss_ss(pred_logits, y, ss_masks_lbl, ss_masks_pred, train=True, smooth_prior=smooth_prior)
        else:
            # no sample selection when calculating loss 
            loss = self._calculate_loss(pred_logits, y, train=True, smooth_prior=smooth_prior)
        
        # training accuracy
        if self.cal_tr:
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y, label_type=f'tr{pi+1}')
                if self.experiment == 'ns_labels' and self.gt_avail:
                    self._get_batch_acc(pred_logits_, y_gt, label_type=f'trg{pi+1}') 
        # training image examples - only log s2 imgs
        if batch_idx==0 and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('tr', x2, pred_logits[1], y_gt=y_gt)
            else:
                self._log_imgs('tr', x2, pred_logits[1], y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
        return loss
    
    
    # def on_train_epoch_end(self):
    #     # get median confidence
    #     conf_tensor = self.training_step_confs[0]/self.training_step_confs[1]
    #     for ti, img_t in enumerate(['s1', 's2']):
    #         for c in range(self.n_classes):
    #             conf_c = conf_tensor[ti,c]
    #             self.log(f'conf_{img_t}_{c}', conf_c, sync_dist=True)
    #     # reinitialize the confidence list
    #     self.training_step_confs = [torch.zeros((2,self.n_classes), dtype=float) for _ in range(2)]
    
    
    # 5. validation step
    def validation_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        x1, x2 = x[:,:self.n_channels_1], x[:,self.n_channels_1:]
        del x
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x1, x2)
        if self.batch_wise:
            # accuracy
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y, label_type=f'v{pi+1}')
                if self.experiment == 'ns_labels' and self.gt_avail:
                    self._get_batch_acc(pred_logits_, y_gt, label_type=f'vg{pi+1}')
        else:
            # confusion matrix
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_conf_matrix(pred_logits_, y, key=f's{pi+1}')
        
        # validation loss
        loss = self._calculate_loss(pred_logits, y, train=False)
        self.log('v_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        # log images
        if batch_idx in [0,5,10] and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('v', x2, pred_logits[1], y_gt=y_gt)
            else:
                self._log_imgs('v', x2, pred_logits[1], y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
        del x1, x2, y, pred_logits
    
    
    def on_validation_epoch_end(self):
        if self.batch_wise:
            pass
        else:
            # summarize confusion matrix
            stats = {}
            for k in self.validation_step_outputs:
                stats[k] = {}
                for sk in self.validation_step_outputs[k]:
                    stats[k][sk] = torch.cat(self.validation_step_outputs[k][sk], dim=0)
                    stats[k][sk] = torch.sum(stats[k][sk], dim=0, keepdim=True)
                # calculate metrics
                self._get_batch_acc_each_epoch(stats[k], label_type='v'+k[1])
            # free memory
            for k in self.validation_step_outputs:
                for sk in self.validation_step_outputs[k]:
                    self.validation_step_outputs[k][sk].clear()  # free memory
    
    
    # 6. test step
    def test_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        x1, x2 = x[:,:self.n_channels_1], x[:,self.n_channels_1:]
        del x
        pred_logits = self.forward(x1, x2)
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y_ns, label_type=f'tsn{pi+1}')
                if self.gt_avail: 
                    y_gt = batch[KEY_GT].squeeze(axis=1)
                    self._get_batch_acc(pred_logits_, y_gt, label_type=f'ts{pi+1}')
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y_gt, label_type=f'ts{pi+1}')
        del x1, x2, pred_logits
        
    
    # auxiliary functions
    # a.0. get ramp-up function
    def _get_rampup_func(self, func_type):
        if func_type=='none':
            rmup_func = lambda x: 1.0
        elif func_type=='linear':
            rmup_func = linear_rampup 
        else:
            rmup_func = sigmoid_rampup
        return rmup_func
            
    # a.1. get model
    def _get_model(self, args):
        if args.resume in ['imagenet','ssl','swsl']: assert args.n_channels==3, \
            'Pretrained encoder weights in smp package only support 3 bands as input!'
        
        if args.fuse_type=='late':
            structure = Unet_lateF_3branch if self.use_com_feat else Unet_lateF_2branch
        elif args.fuse_type=='mid':
            structure = Unet_midF_3branch if self.use_com_feat else Unet_midF_2branch
        mtag = '3' if self.use_com_feat else '2'
        self.net = structure(encoder_name=args.mt,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                             encoder_weights=args.resume if args.resume in ['imagenet','ssl','swsl'] else None,  # pre-trained weights for encoder initialization
                             in_channels_1=self.n_channels_1,    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                             in_channels_2=self.n_channels_2,    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                             classes=self.n_classes,         # model output channels (number of classes in your dataset)
                            )
        print(f"Construct {mtag} encoder branch UNet model with {args.mt} as backbone!")
        
        # load weights from a checkpoint file - tbd
        self.opt_w = None
        if args.resume: 
            if args.resume in ['imagenet','ssl','swsl']:
                print(f"Pre-trained model is loaded from smp package (type:{args.resume})!")
            else:
                assert os.path.isfile(args.resume), 'Given model checkpoint loading path is invalid!'
                # initialize or resume
                if args.prt:
                    checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                    msd = modify_module_state_dict_keys(checkpoint['model_state_dict'])
                    self.net.load_state_dict(msd)
                    print(f"Pre-trained model is loaded from '{args.resume}'!")
                else:
                    print(f"Training is resumed from '{args.resume}'!")
                    # self.opt_w = checkpoint['optimizer_state_dict']
                    # self.last_epoch = checkpoint['epoch']
        else:
            print('No pretrained weights, will start from scratch.')
    
    
    # a.2. get losses
    def _get_loss(self, n_classes, loss_type, if_mask=False):
        # - CE for co-teaching
        self.SSLoss = nn.CrossEntropyLoss(reduction='none', ignore_index=n_classes if if_mask else -100)  
        # - segmentation losses
        self.losses_seg = {}
        # Cross Entropy
        if 'c' in loss_type:
            if n_classes>1:
                if if_mask:
                    self.CELoss = nn.CrossEntropyLoss(ignore_index=n_classes, reduction='none')  
                else:
                    self.CELoss = nn.CrossEntropyLoss(reduction='none')
                print("CrossEntropy loss is used!")
            else:
                self.CELoss = nn.BCEWithLogitsLoss(reduction='none')
                print("BinaryCrossEntropy loss is used!")
            self.losses_seg['c'] = self.CELoss
        else:
            self.CELoss = None  
        # Dice
        if 'd' in loss_type:
            self.DLoss = SoftDiceLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                      from_logits=True, 
                                      ignore_index=n_classes if if_mask else None)
            print("Dice loss is used!")
            self.losses_seg['d'] = self.DLoss
        else:
            self.DLoss = None
        # Focal loss
        if 'f' in loss_type:
            self.FLoss = FocalLoss(mode='multiclass' if n_classes>1 else 'binary', reduction='none',
                                   ignore_index=n_classes if if_mask else None)
            print("Focal loss is used!")
            self.losses_seg['f'] = self.FLoss
        else:
            self.FLoss = None
        assert len(self.losses_seg)>0, 'No segmentation loss is used!'
        
        # - consistency losses
        if self.cl_type == 'mse':
            # MSE loss
            self.CONLoss = nn.MSELoss(reduction='none')
        elif self.cl_type == 'kl':
            # KL divergence
            self.CONLoss = nn.KLDivLoss(reduction='none', log_target=True)
        elif self.cl_type == 'ce':
            if n_classes>1:
                self.CONLoss = nn.CrossEntropyLoss(reduction='none')
            else:
                self.CONLoss = nn.BCEWithLogitsLoss(reduction='none') 
        elif self.cl_type == 'none':
            # assert fuse_type=='mid', f'Consistency loss type is required when the fusion type is {fuse_type}!'
            print('No consistency loss is used!')
        
        
    # a.3. get batch accuracy
    def _get_batch_acc(self, pred_logits, y, label_type='tr'):
        log_batch = self.log_batch if 'tr' in label_type else False
        metrics = evl.evaluate_batch_pl(pred_logits, y, self.n_classes)
        for mk in metrics:
            metr = metrics[mk]
            if torch.prod(torch.tensor(metr.size()))>1:
                for i in range(len(metr)-1):
                    self.log(f'{label_type}_{mk}_{i}', metr[i], on_epoch=True, on_step=log_batch, sync_dist=True)
                self.log(f'{label_type}_m{mk}', metr[-1], on_epoch=True, on_step=log_batch, sync_dist=True)
            else:
                self.log(f'{label_type}_{mk}', metr, on_epoch=True, on_step=log_batch, sync_dist=True)
    
    # a.3-2. get accuracy from confusion matrix after every epoch
    def _get_batch_conf_matrix(self, pred_logits, y, key):
        metrics = evl.get_confuse_matrix_batch_pl(pred_logits, y, self.n_classes)
        for mk in metrics:
            self.validation_step_outputs[key][mk].append(metrics[mk])
    
    def _get_batch_acc_each_epoch(self, stats, label_type='v'):
        metrics = evl.get_batch_acc_from_confuse_matrix(stats, self.n_classes)
        for mk in metrics:
            metr = metrics[mk]
            if torch.prod(torch.tensor(metr.size()))>1:
                for i in range(len(metr)-1):
                    self.log(f'{label_type}_{mk}_{i}', metr[i], sync_dist=True)
                self.log(f'{label_type}_m{mk}', metr[-1], sync_dist=True)
            else:
                self.log(f'{label_type}_{mk}', metr, sync_dist=True)
                
    # a.4. log images
    def _log_imgs(self, prefix, x, pred_logits, y_gt=None, y_ns=None):
        mask_data = pred_logits[0].detach().cpu().numpy().argmax(axis=0)
        class_labels = self.cls_dict
        masks={'prediction': {'mask_data': mask_data, 'class_labels': class_labels},}
        if y_gt is not None:
            masks['ground truth'] = {'mask_data': y_gt[0].detach().cpu().numpy(), 'class_labels': class_labels}
        if y_ns is not None:
            masks['noisy labels'] = {'mask_data': y_ns[0].detach().cpu().numpy(), 'class_labels': class_labels}
        # get rgb img
        if x.shape[1]==3:
            rgb_img = x[0].detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[1]>3:
            rgb_img = x[0,[3,2,1]].detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[1]<3:
            rgb_img = x[0,0].detach().cpu().numpy()
        mask_img = wandb.Image(rgb_img, masks=masks)
        wandb.log({f'{prefix}_masks': mask_img})
        
        
    # a.4. claculate loss
    def _calculate_loss(self, pred_logits, y, smooth_prior=None, train=True):
        # label smoothing
        if self.ls and train:
            assert smooth_prior is not None, 'Smooth prior is required when label smoothing is used!'
            # convert y to one-hot
            y = F.one_hot(y, num_classes=self.n_classes).float().permute((0,3,1,2))
            # label smoothing
            y = self._get_smoothed_labels(y, smooth_prior)
        # calculate losses
        loss = torch.tensor(0.0, device=self.device)
        # segmentation losses
        for pi, pred_logits_ in enumerate(pred_logits):
            loss_branch = torch.tensor(0.0, device=self.device)
            for lk in self.losses_seg:
                L = self.losses_seg[lk]
                if lk in ['c','f']:
                    loss_ = L(pred_logits_, y)
                    loss_ = loss_.mean()
                elif lk in ['d']:
                    loss_ = L(pred_logits_, y)
                loss_branch += loss_
            if train: self.log(f'tr_segloss_{pi+1}', loss_branch, on_epoch=True, on_step=self.log_batch, sync_dist=True)
            loss += loss_branch
        if train: self.log('tr_segloss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        # consistency losses
        if self.cl_type != 'none':
            loss_con = self._calculate_con_loss(pred_logits, train)
            loss += self.cl_weight*loss_con
            if train: self.log('tr_conloss', loss_con, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        if train: self.log('tr_loss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        return loss
    
    def _calculate_loss_ss(self, pred_logits, y, ss_masks_lbl, ss_masks_pred, train=True, smooth_prior=None):
        # label smoothing
        if self.ls and train:
            assert smooth_prior is not None, 'Smooth prior is required when label smoothing is used!'
            # convert y to one-hot
            y = F.one_hot(y, num_classes=self.n_classes).float().permute((0,3,1,2))
            # label smoothing
            y = self._get_smoothed_labels(y, smooth_prior)
            
        # calculate weighted losses
        loss = torch.tensor(0.0, device=self.device)
        # segmentation losses
        for pi, pred_logits_ in enumerate(pred_logits):
            # calculate losses
            loss_branch = torch.tensor(0.0, device=self.device)
            for lk in self.losses_seg:
                L = self.losses_seg[lk]
                if lk in ['c','f']:
                    loss_ = L(pred_logits_, y)
                    loss_ = (loss_*ss_masks_lbl[pi]).sum()/ss_masks_lbl[pi].sum()
                elif lk in ['d']:
                    loss_ = L(pred_logits_, y, weights=ss_masks_lbl[pi])
                # # # # # for debugging # # # # #
                if DEBUG and self.current_epoch<5:
                    print(f"EP{self.current_epoch}|loss_{lk}_{pi+1}: {loss_}")
                # # # # # for debugging # # # # #
                loss_branch += loss_
            if train: self.log(f'tr_segloss_{pi+1}', loss_branch, on_epoch=True, on_step=self.log_batch, sync_dist=True)
            loss += loss_branch
        if train: self.log('tr_segloss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        # consistency losses !!!!!!!TBD
        if self.cl_type != 'none':
            # raise NotImplementedError('Co-teaching consistency loss is not implemented yet!')
            loss_con = self._calculate_con_loss(pred_logits, train, sel_mask=ss_masks_pred)
            loss += self.cl_weight*loss_con
            if train: self.log('tr_conloss', loss_con, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        if train: self.log('tr_loss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        return loss
    
    def _calculate_con_loss(self, pred_logits, train, sel_mask=None):
        # calculate pixel-wise consistency loss
        if not self.use_com_feat:
            if self.cl_type == 'kl':
                pred_logs = [F.log_softmax(pred_logits_, dim=1) for pred_logits_ in pred_logits]
                loss_con1_ = self.CONLoss(pred_logs[0], pred_logs[1].detach())
                loss_con2_ = self.CONLoss(pred_logs[1], pred_logs[0].detach())
            elif self.cl_type == 'ce':
                loss_con1_ = self.CONLoss(pred_logits[0], F.softmax(pred_logits[1].detach(), dim=1))
                loss_con2_ = self.CONLoss(pred_logits[1], F.softmax(pred_logits[0].detach(), dim=1))
            elif self.cl_type == 'mse':
                pred_probs = [F.softmax(pred_logits_, dim=1) for pred_logits_ in pred_logits]
                loss_con1_ = self.CONLoss(pred_probs[0], pred_probs[1].detach())
                loss_con2_ = self.CONLoss(pred_probs[1], pred_probs[0].detach())
        # weighted summation
        if sel_mask is not None:
            if self.cl_type == 'kl' or self.cl_type == 'mse':
                sel_mask = [sel_mask[i][:,None,:,:] for i in range(2)]
            n_sam_1 = sel_mask[0].sum()
            n_sam_2 = sel_mask[1].sum()
            loss_con1 = (loss_con1_*sel_mask[1]).sum()/n_sam_2 if n_sam_2>0 else torch.tensor(0.0, device=self.device)
            loss_con2 = (loss_con2_*sel_mask[0]).sum()/n_sam_1 if n_sam_1>0 else torch.tensor(0.0, device=self.device)
            # # # # # for debugging # # # # #
            if DEBUG and self.current_epoch<5:
                print(f"EP{self.current_epoch}|loss_con1_: {loss_con1_.shape}, sel_mask[0]: {sel_mask[0].shape}, n_sam_1: {n_sam_1}")
                print(f"EP{self.current_epoch}|loss_con2_: {loss_con2_.shape}, sel_mask[1]: {sel_mask[1].shape}, n_sam_2: {n_sam_2}")
                print(f"EP{self.current_epoch}|loss_con1: {loss_con1}, loss_con2: {loss_con2}")
            # # # # # for debugging # # # # #
        else:
            n_sam = pred_logits[0].numel()/self.n_classes
            loss_con1 = loss_con1_.sum()/n_sam
            loss_con2 = loss_con2_.sum()/n_sam
        if train: self.log('tr_conloss_2t', loss_con1, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        if train: self.log('tr_conloss_2i', loss_con2, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        loss_con = loss_con1 + loss_con2
        return loss_con
    
    
    # a.5. label smoothing
    def _get_smoothed_labels(self, y, smooth_prior=None):
        # smooth labels
        if smooth_prior is None:
            # general label smoothing with a given fixed smoothing quantity
            y = y*(1-self.lsf[0]) + self.lsf[0]/self.n_classes
        else:
            # label smoothing with a given prior
            assert y.shape==smooth_prior.shape, 'Shapes of labels and smooth prior are not the same!'
            # case 1: using smooth prior plus a fixed smoothing quantity
            if len(self.lsf)>1:
                if len(self.lsf)>2: 
                    self.lsf = self.lsf[:2]
                    print(f"Only the first two elements of label smoothing factor are used: {self.lsf}")
                # class-wise label smoothing
                y = y*(1-sum(self.lsf)) + smooth_prior*self.lsf[0] + self.lsf[1]/self.n_classes
            # case 2: using smooth prior only
            else:
                y = y*(1-self.lsf[0]) + smooth_prior*self.lsf[0]
        return y
    
    
    # a.6. co-teaching sample selection
    def _get_single_pred_confidence_mask(self, pred_logits):
        # get confidence
        pred_probs = F.softmax(pred_logits.detach(), dim=1)
        pred_logprobs = pred_probs.log()
        # pred_logprobs = F.log_softmax(pred_logits.detach(), dim=1)
        # pred_probs = pred_logprobs.exp()
        if self.ss_ctype == 'ce':
            unc_all = -(pred_probs*pred_logprobs)
            if torch.isnan(unc_all).sum()>0:
                tmp = pred_probs[torch.isnan(unc_all)]
                print(f"EP{self.current_epoch}|Encounter nan in uncertainty: min and max of prob in nan positions are {tmp.min()} - {tmp.max()}")
                unc_all[torch.isnan(unc_all)] = 0
            unc = unc_all.sum(dim=1)
            unc_max = np.log(self.n_classes)
            conf = 1-unc/unc_max
        elif self.ss_ctype == 'gini':
            conf_all = (pred_probs**2)
            if torch.isnan(conf_all).sum()>0:
                tmp = pred_probs[torch.isnan(conf_all)]
                print(f"EP{self.current_epoch}|Encounter nan in confidence: min and max of prob in nan positions are {tmp.min()} - {tmp.max()}")
                conf_all[torch.isnan(conf_all)] = 0
            conf = conf_all.sum(dim=1)
        if torch.isnan(conf).sum()>0:
            tmp = pred_logits*torch.isnan(conf)[:,None]
            tmp = tmp[tmp!=0]
            print(f"EP{self.current_epoch}|nan in final confidence: n-{torch.isnan(conf).sum()}, logits min-{tmp.min()}, logits max-{tmp.max()}")
            conf[torch.isnan(conf)] = 0
        else:
            assert conf.min()>=0 and conf.max()<=1, f"Confidence value is out of range: {conf.min()} - {conf.max()}"    
        return conf
    
    def _get_single_label_confidence_mask(self, pred_logits, y):
        # get confidence
        pred_probs = F.softmax(pred_logits.detach(), dim=1)
        # pred_logprobs = F.log_softmax(pred_logits.detach(), dim=1)
        # pred_probs = pred_logprobs.exp()
        y = F.one_hot(y, num_classes=self.n_classes).float().permute((0,3,1,2))
        conf = (pred_probs*y).sum(dim=1)
        assert conf.min()>=0 and conf.max()<=1, f"Confidence value is out of range: {conf.min()} - {conf.max()}"    
        return conf
    
    def _get_common_enhanced_confidence_masks(self, confidence_masks):
        cmask1, cmask2 = confidence_masks
        commask = cmask1*cmask2
        enhmask1 = (cmask1+commask)/2
        enhmask2 = (cmask2+commask)/2
        enhmasks = [enhmask1, enhmask2]
        return enhmasks
    
    def _get_selection_mask_by_label_confidence(self, conf_mask, y, class_balanced=False):
        # get rumpup sample selection ratio
        if self.current_epoch<=self.ss_nep:
            ss_ratio = 1-self.ss_rmup_func(self.current_epoch, self.ss_nep)*(1-self.ss_prop)
        else:
            ss_ratio = self.ss_prop
        # # # # # for debugging # # # # #
        if DEBUG and self.current_epoch<5:
            print(f"EP{self.current_epoch}|label-seg, ss_ratio={ss_ratio}")
        # # # # # for debugging # # # # #
        if ss_ratio>0 and ss_ratio<1:
            # sample selection
            # weights for the first proportion (ss_ratio) are set to 1, 
            # while the weights for the rest uncertainty ones are set to (0-1) normalized by the maximum confidence in this part
            if not class_balanced:
                # select from all
                nss = int(y.numel()*ss_ratio)
                large_confs, _ = torch.topk(conf_mask, nss, largest=True) # select nss smallest losses
                thrd_conf = large_confs[-1]
                if thrd_conf>0:
                    mask = torch.clip(conf_mask/thrd_conf, min=0., max=1.)
                else:
                    mask = torch.ones_like(conf_mask, device=self.device, requires_grad=False)
            else:
                # select from each class
                mask = torch.zeros_like(conf_mask, device=self.device, requires_grad=False)
                for c in range(self.n_classes):
                    cind_mask = (y==c)
                    nc = cind_mask.sum()
                    nss = int(nc*ss_ratio)
                    if nss>0:
                        c_conf = conf_mask[cind_mask]  # shape: (nc,)
                        c_large_conf, _ = torch.topk(c_conf, nss, largest=True)
                        c_thrd_conf = c_large_conf[-1]
                        if c_thrd_conf>0:
                            c_conf_mask = conf_mask*cind_mask
                            cmask = torch.clip(c_conf_mask/c_thrd_conf, min=0., max=1.)
                            mask += cmask
                        else:
                            mask += cind_mask
            # if self.current_epoch<3:
            #     print(f"EP{self.current_epoch}|{mask.sum()} samples are selected for training!")
        elif ss_ratio==1:
            mask = torch.ones_like(y, device=self.device, requires_grad=False)
        elif ss_ratio==0:
            mask = conf_mask
        else:
            raise ValueError(f"Sample selection ratio is out of range: {ss_ratio}!")
        return mask
    
    
    def _get_selection_mask_by_pred_confidence(self, conf_mask):
        # get rumpup sample selection ratio
        if self.current_epoch<=self.ss_nep:
            ss_ratio = self.ss_rmup_func(self.current_epoch, self.ss_nep)
            mask = (1-ss_ratio)*torch.ones_like(conf_mask, device=self.device, requires_grad=False)+ss_ratio*conf_mask    
        else:
            mask = conf_mask    
        # # # # # for debugging # # # # #
        if DEBUG and self.current_epoch<5:
            print(f"EP{self.current_epoch}|pred-consist, ss_ratio={ss_ratio}")
        # # # # # for debugging # # # # #
        return mask
    
    
    # def _get_selection_mask_by_pred_confidence(self, conf_mask):
    #     # get rumpup sample selection ratio
    #     if self.current_epoch<=self.ss_nep:
    #         ss_ratio = self.ss_rmup_func(self.current_epoch-1, self.ss_nep)
    #     else:
    #         ss_ratio = 1
    #     if ss_ratio>0 and ss_ratio<1:
    #         # sample selection from all
    #         nss = int(conf_mask.numel()*ss_ratio)
    #         large_confs, _ = torch.topk(conf_mask.reshape(-1), nss, largest=True) # select nss smallest losses
    #         thrd_conf = large_confs[-1]
    #         if thrd_conf>0:
    #             mask = torch.clip((conf_mask-thrd_conf)/(large_confs[0]-thrd_conf), min=0., max=1.)
    #         else:
    #             mask = torch.zeros_like(conf_mask, device=self.device, requires_grad=False)
           
    #     elif ss_ratio==1:
    #         mask = conf_mask
    #     elif ss_ratio==0:
    #         mask = torch.zeros_like(conf_mask, device=self.device, requires_grad=False)
    #     else:
    #         raise ValueError(f"Sample selection ratio is out of range: {ss_ratio}!")
    #     return mask
    
    
    # a.7. sample correction by Class-Balanced Self-Training (CBST)
    def _get_correction_mask_by_cbst(self, pred_logits):
        if self.current_epoch<self.sc_inite:
            raise ValueError(f"Sample correction is not started yet! Current epoch: {self.current_epoch}, start epoch: {self.sc_inite}")
        # get correction rate
        if self.current_epoch<=self.sc_inite+self.sc_nep:
            sc_ratio = self.sc_rmup_func(self.current_epoch-self.sc_inite, self.sc_nep)*self.sc_prop
        else:
            sc_ratio = self.sc_prop
        # softmax & initial predicted labels
        pred_probs = F.log_softmax(pred_logits, dim=1).exp()
        pred_confs, pred_lbls = pred_probs.max(dim=1)
        # correction   
        if sc_ratio>0:
            # get class-wise probability thresholds 
            kc = self._get_kc_parameters(pred_confs, pred_lbls, pred_probs, sc_ratio)[None,:, None, None]  # shape: (1, n_classes, 1, 1)
            # if (kc==0).any():
            #     zero_cid = (kc==0).nonzero().squeeze()
            #     for cid in zero_cid:
            #         kc[cid] = pred_probs[:,cid].max()
            #         print(f"EP{self.current_epoch}-{self.current_device}|K of class {cid} = 0, reset to {kc[cid]}")
            # kc = kc[None,:, None, None] # shape: (1, n_classes, 1, 1)
            # get normalized probabilities & new labels
            CBP = pred_probs/kc
            CBPC, CBPL = CBP.max(dim=1)
            # get correction mask
            cmask = CBPC>1
            # get corrected labels
            SCL = F.one_hot(CBPL).permute(0,3,1,2)*cmask[:,None,:,:]
        else:
            cmask = None # torch.zeros_like(pred_lbls, dtype=torch.bool)
            SCL = None
        return cmask, SCL
    
    def _get_kc_parameters(self, pred_confs, pred_lbls, pred_probs, sc_ratio):
        # start_kc = time.time()
        # threshold for each class
        cls_thresh = pred_probs.permute(0,2,3,1).reshape(-1,self.n_classes).max(dim=0).values
        # get class-wise probability thresholds
        for c in range(self.n_classes):
            class_conf = pred_confs[pred_lbls==c]
            if len(class_conf) > 0:
                class_conf = class_conf.sort(descending=True)[0] # return: tuple(sorted, indices)
                len_cls = len(class_conf)
                len_cls_thresh = int(len_cls * sc_ratio)
                if len_cls_thresh != 0:
                    cls_thresh[c] = class_conf[len_cls_thresh-1]
            else:
                print(f"EP{self.current_epoch}-{self.current_device}|No sample for class {c}, set K to {cls_thresh[c]}!")
        # end_kc = time.time()
        # if self.current_epoch<3:
        #     print(f"EP{self.current_epoch}|Time for kc_parameters: {end_kc-start_kc}")
        return cls_thresh
                    
                
            
    