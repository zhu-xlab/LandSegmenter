import os
import sys
import h5py
import wandb
import numpy as np
# appending self-defined package path
sys.path.append('../Codes')

import torch
import torch.nn as nn
from torch import optim

# import segmentation models pytorch package
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# import lightening
import pytorch_lightning as pl

# import model function
from models._get_model import get_model

# self-defined modules
import utils.evaluates as evl
import utils.finetune as ft


KEY_IMG = 'img'
KEY_GT = 'gt'
KEY_NS = 'ns'
KEY_METRICS = ['iou', 'f1', 'precise', 'recall', 'oa']

# sentinel-2 band info
BANDS = {3: [3,2,1],
        #  9: [1,2,3,4,5,6,7,11,12],
         9:  [3,2,1,4,5,6,7,11,12], # satlas/dofa bands
         10: [1,2,3,4,5,6,7,8,11,12],
         12: [0,1,2,3,4,5,6,7,8,9,11,12], # remove B10 - cirrus band
         13: [0,1,2,3,4,5,6,7,8,9,10,11,12]}
WAVELENS = np.array([0.443, 0.490, 0.56, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.940, 1.375, 1.61, 2.19])
# wavelengths = [0.665, 0.56, 0.49, 0.705, 0.74, 0.783, 0.842, 1.61, 2.19] # DOFA order
WAVELENS_S1 = [3.75, 3.75]

class UNET_PL_FT(pl.LightningModule):
    # 1. training setup
    def __init__(self, args):
        super().__init__()
        # pl.seed_everything(args.seed)
        self.last_epoch = 0
        self.max_epochs = args.epochs
        self.batch_wise = False  # whether calculate metrics batch-wisely or as a whole
        # # # # model architecture # # # #
        self.n_classes = args.n_classes
        self.n_channels = args.n_channels
        self.use_dofa = 'dofa' in args.prt_tag
        self._get_model(args)

        # # # # segmentation loss # # # #
        self._get_loss(args.loss_type, args.n_classes, args.if_mask)
        
        # # # # other hyperparameters # # # #
        self.experiment = 'gt_labels'  # used for pre-training, fixed here for fine-tuning
        self.gt_avail = True           # used for pre-training, fixed here for fine-tuning
        self.opt = args.opt
        self.if_mask = args.if_mask
        self.log_batch = args.batch_to_wandb
        self.cal_tr = args.cal_tr
        self.current_device = args.current_device
        self.display_inv = args.display_inv
        self.cls_dict = args.cls_dict
        # for fine-tuning
        self.eft_tag = args.eft_tag
        self.dft_tag = args.dft_tag
        self.eft_ep = args.eft_ep
        # self.eft_lr = args.eft_lr
        self.lr = args.lr         # initial learning rate
        self.lr_vf = args.lr_vf   # varying factor of lr => lr*lr_vf**i
        self.schdt = args.schdt   # type of scheduler
        self.schdv = args.schdv   # value(s) of scheduler
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        
        # for recording tp, fp, fn, tn in on_validation_epoch_end
        if not self.batch_wise: 
            self.validation_step_outputs = {'tp': [], 'fp': [], 'fn': [], 'tn': []}
            self.test_step_outputs = {'tp': [], 'fp': [], 'fn': [], 'tn': []}
        
        # for test data
        self.only_test = args.only_test
        self.test_preds = []
        self.test_dir = args.pdir
        
        
    # 2. forward propagation
    def forward(self, x):
        if self.use_dofa:
            if self.n_channels==2:  # for Sentinel-1
                out = self.net(x, wave_list=WAVELENS_S1).squeeze(axis=1)
            else:   # for Sentinel-2
                out = self.net(x, wave_list=WAVELENS[BANDS[self.n_channels]]).squeeze(axis=1)
        else:
            out = self.net(x).squeeze(axis=1)
        return out
    
    
    # 3. Optimizer
    def configure_optimizers(self):
        if self.dft_tag == 'dsam':
            params = self.parameters()
        else:
            assert self.lr_vf<1 and self.lr_vf>0, 'Please provide correct varying factor for lr!'
            params = ft.set_adaptive_lr_for_decoder_layers(self.net, self.lr, self.dft_tag[1:], self.lr_vf)
        # optimizer
        if self.opt=='adam':
            optimizer = optim.Adam(params, lr=self.lr, weight_decay=1e-8)
            print('Adam is used as optimizer!')
        elif self.opt=='sgd':
            optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)
            print('SGD is used as optimizer!')
        else:
            raise ValueError("Please provide correct optimizer type (adam or sgd)!")
        # scheduler
        if self.schdt=='none':
            return optimizer
        else: 
            assert len(self.schdv)>0, 'Please provide correct scheduler values!'
            if self.schdt=='step':
                print(f'Initial LR (={self.lr}) is scheduled to reduce at the {self.schdv} epochs.')
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.schdv, gamma=0.5)
            elif self.schdt=='cos':
                T = int(self.schdv[0])
                print(f'T:{T}')
                if len(self.schdv)>=2:
                    final_lr = self.schdv[1]
                    if len(self.schdv)>2:
                        print(f'More values than two ({self.schdv}) are provided for cosine annealing scheduler, only the first two ({T, final_lr}) are used!')
                else:
                    final_lr = 0.00001
                print(f'Initial LR (={self.lr}) is scheduled to reduce to {final_lr} in {T} epochs.')
                if T < self.max_epochs:
                    lambda_func = lambda epoch: (final_lr+(1+np.cos(np.pi*(epoch/T)))*(self.lr-final_lr)/2)/self.lr if epoch<T else final_lr/self.lr
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
                    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [sheduler1, scheduler0], milestones=[T])
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T, eta_min=final_lr)
            elif self.schdt=='exp':
                gamma = self.schdv[0]
                if len(self.schdv)>1:
                    print(f'More values than one ({self.schdv}) are provided for exponential scheduler, only the first one ({gamma}) is used!')
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    
    # 4. training step
    def training_step(self, batch, batch_idx):
        # reset learning rate and unfreeze encoder weights for encoder (the whole net) finetuning
        if self.eft_tag=='eft' and self.current_epoch==self.eft_ep-1:
            self.net = ft.unfix_model_weights(self.net)
            print(f'Encoder weights are unfixed at Epoch {self.eft_ep}!')
            # ft.reset_learning_rate(self.optimizers(), self.eft_lr)
        # main step
        x = batch[KEY_IMG]
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x).squeeze(axis=1)
        if self.if_mask: 
            discard_masks = batch['mask_discard'].squeeze(axis=1)
            y[discard_masks!=0] = self.n_classes
        # calculate losses
        loss = torch.tensor(0.0, device=self.device)
        for lk in self.losses:
            L = self.losses[lk]
            loss_ = L(pred_logits, y)
            loss += loss_
            self.log(f'tr_loss_{lk}', loss_, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        self.log('tr_loss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        # training accuracy
        if self.cal_tr:
            self._get_batch_acc(pred_logits, y, label_type='tr')
            if self.experiment == 'ns_labels' and self.gt_avail:
                self._get_batch_acc(pred_logits, y_gt, label_type='trg') 
        # training image examples  
        if batch_idx==0 and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('tr', x, pred_logits, y_gt=y_gt)
            else:
                self._log_imgs('tr', x, pred_logits, y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
        return loss
    
    
    # 5. validation step
    def validation_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x).squeeze(axis=1)
        if self.batch_wise:
            # accuracy
            self._get_batch_acc(pred_logits, y, label_type='v')
            if self.experiment == 'ns_labels' and self.gt_avail:
                self._get_batch_acc(pred_logits, y_gt, label_type='vg')
        else:
            # confusion matrix
            self._get_batch_conf_matrix(pred_logits, y)
        # log images
        if batch_idx in [0,5,10] and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('v', x, pred_logits, y_gt=y_gt)
            else:
                self._log_imgs('v', x, pred_logits, y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)


    def on_validation_epoch_end(self):
        if self.batch_wise:
            pass
        else:
            # summarize confusion matrix
            stats = {}
            for k in self.validation_step_outputs:
                stats[k] = torch.cat(self.validation_step_outputs[k], dim=0)
                stats[k] = torch.sum(stats[k], dim=0, keepdim=True)
            # calculate metrics
            self._get_batch_acc_each_epoch(stats, label_type='v')
            # free memory
            for k in self.validation_step_outputs:
                self.validation_step_outputs[k].clear()  # free memory
    
    
    # 6. test step
    def test_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x).squeeze(axis=1)
        # calculate test accuracies from all the predictions
        # confusion matrix
        self._get_batch_conf_matrix(pred_logits, y, if_val=False)
        # save images
        if torch.cuda.is_available():
            preds = pred_logits.detach().float().cpu().numpy().argmax(axis=1).astype(np.uint8)
        else:
            preds = pred_logits.detach().float().numpy().argmax(axis=1).astype(np.uint8)
        self.test_preds.append(preds)
    
    
    def on_test_epoch_end(self):
        # summarize confusion matrix
        stats = {}
        for k in self.test_step_outputs:
            stats[k] = torch.cat(self.test_step_outputs[k], dim=0)
            stats[k] = torch.sum(stats[k], dim=0, keepdim=True)
        # calculate metrics
        log_opt = False if self.only_test else True
        test_metrics = self._get_batch_acc_each_epoch(stats, log_to_wandb=log_opt)
        # concatenate test predictions
        test_preds = np.concatenate(self.test_preds, axis=0)
        print(f'Shape of test_preds:{test_preds.shape}')
        with h5py.File(os.path.join(self.test_dir,'test_preds.h5'), 'w') as f:
            f.create_dataset('preds', data=test_preds)
        # output statistics
        for mk in test_metrics:
            metr = test_metrics[mk]
            if torch.prod(torch.tensor(metr.size()))>1:
                for i in range(len(metr)-1):
                    print(f'{mk}_{i}: {metr[i]}')
                print(f'm{mk}: {metr[-1]}')
            else:
                print(f'{mk}: {metr}')
        
    
    # auxiliary functions
    # a.1. get model
    def _get_model(self, args):
        self.net = get_model(args.frm, args.mt, self.n_classes, args.n_channels, args.prt_tag, 
                             patch_size=args.patch_size, im_size=args.im_size,
                             weight_path=args.wpath, weight_load_tag=args.wload_tag, 
                             enc_finetune_tag=args.eft_tag)

    # a.2. get losses
    def _get_loss(self, loss_type, n_classes, if_mask):
        self.losses = {}
        # Cross Entropy
        if 'c' in loss_type:
            if n_classes>1:
                if if_mask:
                    self.CELoss = nn.CrossEntropyLoss(ignore_index=n_classes)  
                else:
                    self.CELoss = nn.CrossEntropyLoss()
                print("CrossEntropy loss is used!")
            else:
                self.CELoss = nn.BCEWithLogitsLoss()
                print("BinaryCrossEntropy loss is used!")
            self.losses['c'] = self.CELoss
        else:
            self.CELoss = None  
        # Dice
        if 'd' in loss_type:
            self.DLoss = DiceLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                  from_logits=True, 
                                  ignore_index=n_classes if if_mask else None)
            print("Dice loss is used!")
            self.losses['d'] = self.DLoss
        else:
            self.DLoss = None
        # Focal loss
        if 'f' in loss_type:
            self.FLoss = FocalLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                   ignore_index=n_classes if if_mask else None)
            print("Focal loss is used!")
            self.losses['f'] = self.FLoss
        else:
            self.FLoss = None
        assert len(self.losses)>0, 'No loss is used!'
        
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

    def _get_batch_conf_matrix(self, pred_logits, y, if_val=True):
        metrics = evl.get_confuse_matrix_batch_pl(pred_logits, y, self.n_classes)
        for mk in metrics:
            if if_val:
                self.validation_step_outputs[mk].append(metrics[mk])
            else:
                self.test_step_outputs[mk].append(metrics[mk])
    
    def _get_batch_acc_each_epoch(self, stats, label_type='v', log_to_wandb=True):
        metrics = evl.get_batch_acc_from_confuse_matrix(stats, self.n_classes)
        if log_to_wandb:
            for mk in metrics:
                metr = metrics[mk]
                if torch.prod(torch.tensor(metr.size()))>1:
                    for i in range(len(metr)-1):
                        self.log(f'{label_type}_{mk}_{i}', metr[i], sync_dist=True)
                    self.log(f'{label_type}_m{mk}', metr[-1], sync_dist=True)
                else:
                    self.log(f'{label_type}_{mk}', metr, sync_dist=True)
        else:
            return metrics
    
    # a.4. log images
    def _log_imgs(self, prefix, x, pred_logits, y_gt=None, y_ns=None):
        if str(self.device)=='cpu':
            mask_data = pred_logits[0].detach().float().numpy().argmax(axis=0)
        else:
            mask_data = pred_logits[0].detach().cpu().numpy().argmax(axis=0)
        class_labels = self.cls_dict
        masks={'prediction': {'mask_data': mask_data, 'class_labels': class_labels},}
        if y_gt is not None:
            masks['ground truth'] = {'mask_data': y_gt[0].detach().cpu().numpy(), 'class_labels': class_labels}
        if y_ns is not None:
            masks['noisy labels'] = {'mask_data': y_ns[0].detach().cpu().numpy(), 'class_labels': class_labels}
        if x.shape[1]==3:
            mask_img = wandb.Image(x[0].detach().cpu().numpy().transpose((1,2,0)), masks=masks)
        elif x.shape[1]>3:
            mask_img = wandb.Image(x[0,[3,2,1]].detach().cpu().numpy().transpose((1,2,0)), masks=masks)
        elif x.shape[1]<3:
            mask_img = wandb.Image(x[0,0].detach().cpu().numpy(), masks=masks)
        else:
            print(f'Please provide correct number of channels for image logging! Current number of channels is {x.shape[1]}.')
        wandb.log({f'{prefix}_masks': mask_img})
        
    # # a.4. define wandb metrics
    # def _define_wandb_metrics(self):
    #     log_metrics = ['tr_loss']+[f'tr_loss_{lk}' for lk in self.losses]+\
    #                     [f'{lb}_{metr}_{c}' for lb in ['tr','trg','v','vg','ts','tsn'] for metr in ['iou','f1','precise','recall'] for c in range(self.n_classes)]+\
    #                     [f'{lb}_m{metr}' for lb in ['tr','trg','v','vg','ts','tsn'] for metr in ['iou','f1','precise','recall']]+\
    #                     [f'{lb}_oa' for lb in ['tr','trg','v','vg','ts','tsn']]
    #     for lm in log_metrics:
    #         wandb.define_metric(lm, summary='mean')
    
    
                

