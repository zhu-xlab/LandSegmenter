# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import wandb
import random

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
# from PIL.Image import Image
from einops import rearrange

# import lightening
import pytorch_lightning as pl
from torch import optim, nn
from torch.nn import functional as F

# from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from kornia.losses import (                   # type: ignore
    dice_loss, binary_focal_loss_with_logits, focal_loss,
    FocalLoss, DiceLoss
)
from sam2.utils.myLosses import SoftDiceLoss

# import sam2 package
# from sam2.modeling.sam2_base import SAM2Base
# from sam2.utils.transforms import SAM2Transforms
from sam2.build_sam import build_sam2
from sam2.modeling.text_simple_tokenizer import SimpleTokenizer
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer

from dataset.datasets_settings import class_names, class_weight_list    

FINAL_LR = 1e-6
KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
KEY_WEAKLBL = 'weak'
MASK_CODE = 255
INPUT_SIZE = 1024
TEXT_PREFIX = 'A photo of '


############################### FOR USFS LC ################################
MULTI_MASK_KEYS = {'usfs': ['usfs-lc', 'usfs-lu'],
                   'nlcd': ['nlcd-lc', 'nlcd-imp']}

class SAM2ImageTrainerFT(BaseTrainer):
    def __init__(
        self,
        args,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        args.use_multiclass_loss = True
        if args.use_aux_decoder:
            args.use_aux_decoder = False
            print("Fine-tuning does not use auxiliary decoder. Set use_aux_decoder=False")
        super().__init__(args)
        self.fine_tune_acc = args.fine_tune_acc
        self.use_binary_loss = args.use_binary_loss
        self.use_class_multiplier = args.use_class_multiplier
        # self.save_path = args.save_path
        # place holders for calculating validation accuracies
        # for gt
        self.ys, self.masks, self.ious = {}, {}, {}
        for k in ['val', 'ts']:
            self.ys[k], self.masks[k], self.ious[k] = [], [], []
        # self.ts_iou, self.ts_iou_pred = [], []
        # self.ts_acc, self.ts_acc_pred = [], []
        # self.ts_exist = []
        if args.ft_wpath is None or args.ft_wpath=='None':
            self._freeze_backbones() 
        else:
            self._load_model(args.ft_wpath)
            if args.freeze_backbones_only:
                self._freeze_backbones()
            else:
                self._freeze_encoder()
        
    
    def _load_model(self, checkpoint_path):
        # rename keys
        ckpt = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        ckpt_cp = ckpt.copy()
        for k in ckpt_cp:
            if 'aux' in k: 
                del ckpt[k]
            else:
                if k.startswith('model.'): ckpt[k[6:]] = ckpt.pop(k)
        # load weights
        self.model.load_state_dict(ckpt, strict=True)
    

    def _freeze_encoder(self):
        # freeze encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
    

    def _freeze_backbones(self):
        for name, param in self.model.image_encoder.named_parameters():
            if 'prompt_generator' in name:
                if 'dofa' in name:
                    param.requires_grad = False
                else:
                    continue
            else:
                param.requires_grad = False


    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        # main step
        # get data
        x, y, waves, y_exist, n_class, prompt_token, if_weak_lbl, name_key = \
            self._get_data(batch, return_single=True)
        
        # # # major forward process # # #           
        # image embeddings
        img_feats = self._get_image_embed(x, waves, False)

        # text embeddings
        text_embeddings = self._get_text_embeddings(prompt_token)

        # prediction
        masks, iou_predictions, precise_predictions, exist_predictions = \
            self.forward_decoder(img_feats, text_embeddings, n_class, x.shape[0])
        if x.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(masks, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # prediction accuracies
        if self.fine_tune_acc:
            precises, ious = self._get_accs(masks.detach(), y)
        else:
            precises, ious, y_exist =  None, None, None
            
        # calculate losses 
        class_weights = self.class_weight_tensors[name_key] if self.use_class_weights else None
        loss = self._calculate_loss(n_class, y, y_exist, ious, precises, 
                                    masks, exist_predictions, iou_predictions, precise_predictions,
                                    class_weights=class_weights, log=False,
                                    )

        # training image examples  
        if batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
            self._log_imgs(f'tr_{name_key}', x[0], masks[0], y[0])
        
        # log loss
        self.log('tr_loss', loss, on_epoch=True, on_step=False, sync_dist=True) #, batch_size=x.shape[0])
        
        return loss


    def on_train_epoch_start(self):
        # class weights
        if self.current_epoch==0 and self.use_class_weights:
            self.class_weight_tensors = \
                {k:torch.tensor(class_weight_list[k], device=self.device) for k in class_weight_list}
            if self.class_weights_scale!=10:
                assert self.class_weights_scale>1 and self.class_weights_scale<10, 'class_weight_scale should be larger than 1 and smaller than 10!'
                for k in self.class_weight_tensors:
                    self.class_weight_tensors[k] = (self.class_weight_tensors[k]*(self.class_weights_scale-1)-self.class_weights_scale+10)/9
            if self.trainer.global_rank==0:
                print(f'Class weights are used with scale={self.class_weights_scale}!')
                print(f'Class weights: {self.class_weight_tensors}')
        # print(f'Starting epoch {self.current_epoch}')
        # return super().on_train_epoch_start()
    

    def validation_step(self, batch, batch_idx):
        self._gather_predictions_from_batch(batch, batch_idx, key='val')

    
    def on_validation_epoch_end(self):
        oa, ious_all, ious_single = self._calculate_acc_at_epoch_end(key='val')
        self.log('val_oa', oa, on_epoch=True, on_step=False, sync_dist=True)
        for i, iou in enumerate(ious_all):
            if i<len(ious_all)-1:
                self.log(f'val_ioua_{i}', iou, on_epoch=True, on_step=False, sync_dist=True)
            else:
                self.log('val_mioua', iou, on_epoch=True, on_step=False, sync_dist=True)
        for i, iou in enumerate(ious_single):
            if i<len(ious_single)-1:
                self.log(f'val_ious_{i}', iou, on_epoch=True, on_step=False, sync_dist=True)
            else:
                self.log('val_mious', iou, on_epoch=True, on_step=False, sync_dist=True)


    # ------ Model utilities ------ #
    def _calculate_loss(self, n_classes, y, y_exist=None, ious=None, precises=None,
                        mask_predictions=None, exist_predictions=None, 
                        iou_predictions=None, precise_predictions=None,
                        class_weights=None, log=True):

        # calculate losses 
        loss = torch.tensor(0.0, device=self.device)
        # no-value mask
        mask = (y!=MASK_CODE) # no data mask
        mask = mask.unsqueeze(1).expand_as(mask_predictions)
        y_one_hot = F.one_hot(y.clamp(0, n_classes - 1), num_classes=n_classes).permute(0, 3, 1, 2)  # Reshape to [5, 3, 4, 4]
        y_one_hot = y_one_hot * mask
        y_one_hot_wnodata = y_one_hot * mask + (~mask) * MASK_CODE

        # binary mask loss
        if self.use_binary_loss:
            for w, k in zip([self.wdice, self.wfocal, self.wce], ['dice', 'focal', 'bce']):
                if w>0:
                    if k=='bce':
                        loss_ = w*F.binary_cross_entropy_with_logits(
                            mask_predictions, y_one_hot.float(), 
                            weight=mask.float(), 
                            pos_weight=class_weights[:, None, None] if class_weights is not None else None, 
                            reduction='none',
                        )
                        loss_ = loss_[mask].mean()
                    elif k=='dice':
                        loss_ = w*dice_loss(
                            mask_predictions, y, 
                            average='macro', weight=class_weights,
                            ignore_index=MASK_CODE
                        )
                    elif k=='focal':
                        loss_ = w*binary_focal_loss_with_logits(
                            mask_predictions, y_one_hot_wnodata,
                            reduction='none', pos_weight=class_weights,
                            ignore_index=MASK_CODE,
                        )
                        loss_ = loss_[mask].mean()
                    loss = loss + loss_*n_classes if self.use_class_multiplier else loss + loss_

        # multi-class loss
        if class_weights is None:
            for L, w in zip([self.DLoss, self.FLoss, self.CELoss], [self.wdice, self.wfocal, self.wce]):
                if w>0:
                    loss_ = w*L(mask_predictions, y.clone())  # otherwise encounter the error: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.LongTensor [6, 256, 256]] is at version 2; expected version 0 instead.
                    loss = loss+loss_
                else:
                    continue
        else:
            # pixel_weights_mean = weights.mean(axis=1)
            for w, k in zip([self.wdice, self.wfocal, self.wce], ['dice', 'focal', 'ce']):
                if k == 'ce':
                    loss_ = F.cross_entropy(mask_predictions, y, ignore_index=MASK_CODE, reduction='none')
                    loss_ = w*(loss_*class_weights)[mask[:,0]].mean() # loss_: [batch_size, h, w]
                elif k == 'dice':
                    loss_ = w*self.SoftDLoss(mask_predictions, y, weights=class_weights)
                elif k == 'focal':
                    loss_ = focal_loss(mask_predictions, y, reduction='none', ignore_index=MASK_CODE, alpha=0.25)
                    loss_ = loss_.mean(axis=(1))   # loss_: [batch_size, c, h, w]
                    loss_ = w*(loss_*class_weights)[mask[:,0]].mean()
        if log: self.log('tr_loss_mask', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        # existance loss
        if y_exist is not None:
            loss_exist_matrix = self.BCELoss_noreduce(exist_predictions, y_exist)
            # calculate the proportion of labeled pixels to serve as the weights
            y_proportions = ((y!=MASK_CODE).sum(axis=(1,2))/(y.shape[1]*y.shape[2]))[:,None]
            y_proportions = y_proportions.repeat(1, n_classes)
            w_exist = torch.max(y_proportions, y_exist)
            # weighted existance loss according to the reliability
            loss_exist = (w_exist*loss_exist_matrix).sum()/loss_exist_matrix.numel()
            loss_exist = loss_exist * n_classes
            if log: self.log('tr_loss_exist', loss_exist, on_epoch=True, on_step=False, sync_dist=True)
        else:
            loss_exist = torch.tensor(0.0, device=self.device)

        # accuracy losses
        if ious is not None:
            loss_iou_matrix = self.L1Loss(iou_predictions, ious)
            # calculate weighted iou loss
            # weighting strategy: considering 1. the reliability of the existance prediction, 2. the reliability of the iou prediction
            loss_iou = self._calculate_acc_loss(loss_iou_matrix, y_proportions, y_exist)
        else:
            loss_iou = torch.tensor(0.0, device=self.device)
        if precises is not None:
            loss_precise_matrix = self.L1Loss(precise_predictions, precises)
            loss_precise = self._calculate_acc_loss(loss_precise_matrix, y_proportions, y_exist)
        else:
            loss_precise = torch.tensor(0.0, device=self.device)

        # final loss
        loss = loss \
            + loss_iou*n_classes \
            + loss_precise*n_classes \
            + loss_exist
        if log: self.log('tr_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    

    def _gather_predictions_from_batch(self, batch, batch_idx, key):
        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens, _, name_key = self._get_data(batch)

        # prediction
        with torch.no_grad():
            masks, iou_predictions, precise_predictions, existance_predictions, _, _ = \
                self.forward(x, prompt_tokens, waves)
        if x.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(masks, size=x.shape[-2:], mode='bilinear', align_corners=False)
        masks = masks.detach()
        mask_log = masks[0]

        # ious
        _, ious = self._get_accs(masks, y)
        masks = masks.argmax(dim=1)

        # record results
        self.ys[key].append(y.cpu().numpy().astype(np.uint8))
        self.masks[key].append(masks.cpu().numpy().astype(np.uint8))
        self.ious[key].append(ious.cpu().numpy())

        if batch_idx<2 and self.trainer.global_rank==0:
            self._log_imgs(f'val_{name_key}', x[0], mask_log, y[0])
    

    def _calculate_acc_at_epoch_end(self, key):
        # patch-wise accuracies
        ious_single = np.concatenate(self.ious[key], axis=0)
        n_class = ious_single.shape[1]
        self.ious[key].clear()
        ious_single = np.mean(ious_single, axis=0)
        ious_single = np.concatenate([ious_single, np.mean(ious_single, keepdims=True)])

        # calculate iou from all samples
        ys = np.concatenate(self.ys[key], axis=0)
        self.ys[key].clear()
        ys = np.where(ys!=MASK_CODE, ys+1, 0)
        masks = np.concatenate(self.masks[key], axis=0)+1
        self.masks[key].clear()
        masks[ys==0] = 0
        ious_all, oa = self._calculate_acc_from_all(masks, ys, n_class)

        return oa, ious_all, ious_single


