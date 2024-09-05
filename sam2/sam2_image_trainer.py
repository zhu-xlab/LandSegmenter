# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
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

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import albumentations as A

# import sam2 package
# from sam2.modeling.sam2_base import SAM2Base
# from sam2.utils.transforms import SAM2Transforms
from sam2.build_sam import build_sam2
from sam2.modeling.text_simple_tokenizer import SimpleTokenizer


FINAL_LR = 1e-8
KEY_IMG = 'image'
KEY_LBL = 'label'
# KEY_CLSN = 'cn_dict'
KEY_LBLTYPE = 'label_type'
MASK_CODE = 255
INPUT_SIZE = 1024

CLS_DICT = {0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub and Scrub', 6:'Built area', 7:'Bare land', 8:'Ice and Snow'}


class SAM2ImageTrainer(pl.LightningModule):
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
        super().__init__()
        # copnstruct sam2-based model (image encoder and prompt encoder weights are automatically fixed)
        self.model = build_sam2(args.config_file, args.ckpt_path, args.ckpt_path_clip)
        # set up the losses: dice and focal
        self._get_loss()

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        # others TO BE DEFINED
        self.multimask_output = False
        self.tokenizer = SimpleTokenizer()
        self.n_prompts = args.n_prompts if type(args.n_prompts)==type([]) else [args.n_prompts]
        # loss weights
        self.wfocal = args.wfocal
        self.wdice = args.wdice
        # optimizer
        self.opt = args.opt
        self.lr = args.lr
        # scheduler
        self.schdt = args.schdt
        self.schdv = args.schdv
        self.max_epochs = args.epochs
        # display interval
        self.display_inv = args.display_inv
        

    # ------ forward propogation ------ #
    def forward(self, x, prompt):  # text prompt: here is the tokenized tensor
        batch_size = x.shape[0]
        # image embeddings
        img_feats = self._get_image_embed(x)
        # prompt text embeddings
        n_classes = prompt.shape[0]
        text_embeddings = self.model.sam_prompt_encoder(prompt) # [n_prompt, dim] dim=512 (clip dim)
        text_embeddings = self.model.sam_prompt_fn(text_embeddings) # [n_prompt, dim] dim=256
        text_embeddings = rearrange(text_embeddings.repeat(batch_size, 1),
                                    '(batch_size n_classes) dim -> (n_classes batch_size) dim', 
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        # Predict masks
        high_res_features = [
            feat_level.repeat(n_classes, 1, 1, 1) for feat_level in img_feats["high_res_feats"]
        ]
        masks, iou_predictions = self.model.sam_mask_decoder(
            image_embeddings=img_feats["image_embed"].repeat(n_classes, 1, 1, 1),
            multimask_output=self.multimask_output,
            text_embeddings=text_embeddings.unsqueeze(1),
            high_res_features=high_res_features,
        )

        masks = (torch.clamp(masks, -32.0, 32.0) + 32.0)/64.0
        masks = rearrange(torch.squeeze(masks, dim=1), 
                          '(n_classes batch_size) H W -> batch_size n_classes H W',
                          batch_size=batch_size,
                          n_classes=n_classes)
        iou_predictions = rearrange(torch.squeeze(iou_predictions, dim=1), 
                                    '(n_classes batch_size) -> batch_size n_classes',
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        return masks, iou_predictions
    

    # ------ Optimizer ------ #
    def configure_optimizers(self):
        # optimizer
        if self.opt=='adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
            print('Adam is used as optimizer!')
        elif self.opt=='adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)  # weight_decay is set according to sam2
            print('AdamW is used as optimizer!')
        elif self.opt=='sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            print('SGD is used as optimizer!')
        else:
            raise ValueError("Please provide correct optimizer type (adam or sgd)!")
    
        # scheduler
        if self.schdt=='none':
            return optimizer
        else: 
            assert len(self.schdv)>0, 'Please provide correct scheduler values!'
            if self.schdt=='step':
                print(f'Initial LR (={self.lr}) is scheduled by {self.schdt} to reduce at the {self.schdv} epochs.')
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.schdv, gamma=0.5)
            elif self.schdt=='cos':
                T = int(self.schdv[0])
                if len(self.schdv)>=2:
                    final_lr = self.schdv[1]
                    if len(self.schdv)>2:
                        print(f'More values than two ({self.schdv}) are provided for cosine annealing scheduler, only the first two ({T, final_lr}) are used!')
                else:
                    final_lr = FINAL_LR
                print(f'Initial LR (={self.lr}) is scheduled by {self.schdt} to reduce to {final_lr} in {T} epochs.')
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
                print(f'Initial LR (={self.lr}) is scheduled by {self.schdt} with gamma={gamma}.')
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
    

    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        # main step
        # get data
        x = batch[KEY_IMG]
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).contiguous()
        y = batch[KEY_LBL]
        prompt_tokens, y, n_classes = self._prepare_prompt(CLS_DICT, y)
        # lbl_type = batch[KEY_LBLTYPE]  # indicate the type of labels: 0 - gt, 1 - ns
        # assert lbl_type in [0, 1], 'Please provide correct label type (0 - gt, 1 - ns)!'
        
        # prediction
        masks, iou_predictions = self.forward(x, prompt_tokens)
        
        # prepare labels
        # if lbl_type==1: # ns labels
        #     y1 = y.copy()
        #     y2 = y
        # else: # gt labels
        #     y1 = y2 = y
        # calculate losses 
        loss = torch.tensor(0.0, device=self.device)
        # # multi-class loss
        # for L, w in zip([self.DLoss, self.FLoss], [self.wdice, self.wfocal]):
        #     loss_ = w*L(masks, y)
        #     loss += loss_
        # self.log('tr_loss_multic', loss, on_epoch=True, on_step=False, sync_dist=True)
        # binary loss
        for L, w in zip([self.BDLoss, self.BFLoss], [self.wdice, self.wfocal]):
            for c in range(n_classes):  
                y_ = (y==c).to(y.dtype)
                y_[y==MASK_CODE] = MASK_CODE
                loss_ = w*L(masks[:,c], y_)
                loss += loss_
        self.log('tr_loss_mask', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        # iou loss
        ious = self._get_ious(masks.detach(), y)
        loss += self.L1Loss(iou_predictions, ious)*n_classes
        self.log('tr_loss_', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        # training image examples  
        if batch_idx==0 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
            self._log_imgs('tr', x[0], masks[0], y[0])
        return loss
    

    def validation_step(self, batch, batch_idx):
        # get data
        x = batch[KEY_IMG]
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).contiguous()
        y = batch[KEY_LBL]
        # prompt = batch[KEY_PROMPT]
        prompt_tokens, y, n_classes = self._prepare_prompt(CLS_DICT, y, train_mode=False)
        
        # prediction
        masks, iou_predictions = self.forward(x, prompt_tokens)
        
        # ious
        ious = self._get_ious(masks.detach(), y)
        self.log('val_mious', ious.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mious_pred', iou_predictions.mean(), on_epoch=True, on_step=False, sync_dist=True)
        if batch_idx==0 and self.trainer.global_rank==0:
            self._log_imgs('val', x[0], masks[0], y[0])


    def test_step(self, batch, batch_idx):
        # get data
        x = batch[KEY_IMG]
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).contiguous()
        y = batch[KEY_LBL]
        # prompt = batch[KEY_PROMPT]
        prompt_tokens, y, n_classes = self._prepare_prompt(CLS_DICT, y, train_mode=False)
        
        # prediction
        masks, iou_predictions = self.forward(x, prompt_tokens)

        # ious
        ious = self._get_ious(masks.detach(), y)
        print('ious:', ious)
        print('ious mean:', ious.mean())
        self.log('test_mious', ious.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_mious_pred', iou_predictions.mean(), on_epoch=True, on_step=False, sync_dist=True)
        if batch_idx==0 and self.trainer.global_rank==0:
            self._log_imgs('test', x[0], masks[0], y[0])


    # ------ Model utilities ------ #
    def _get_image_embed(
        self,
        input_image: Union[torch.Tensor],  # [B, C, H, W]
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        batch_size = input_image.shape[0]
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}  # self._features
        return features
    

    def _prepare_prompt(self, class_name_dict, lbl, train_mode=True):
        """ Tokenization of string inputs for the model. """
        truncate = False
        context_length = self.model.sam_prompt_encoder.context_length

        # generate prompt texts and reprepare labels
        nc_total = len(class_name_dict)
        if train_mode:
            n_prompt = random.choice(self.n_prompts)
            if n_prompt == nc_total:
                texts = [class_name_dict[i] for i in range(nc_total)]
                gt = lbl
            else:
                lbl_classes = torch.unique(lbl).tolist()
                lbl_classes.remove(255)
                if n_prompt > len(lbl_classes):
                    prompt_inds_empty = [i for i in range(nc_total) if i not in lbl_classes]
                    lbl_classes += random.sample(prompt_inds_empty, n_prompt-len(lbl_classes))
                prompt_inds = random.sample(lbl_classes, n_prompt)
                texts = [class_name_dict[i] for i in prompt_inds]
                # prepare labels
                gt = torch.full_like(lbl, MASK_CODE)
                for i, ind in enumerate(prompt_inds):
                    gt[lbl==ind] = i    
        else:    # validation and test modes taking all the class names
            n_prompt = nc_total
            texts = [class_name_dict[i] for i in range(nc_total)]
            gt = lbl

        # Tokenization of string inputs for the model
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long, device=lbl.device)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens, device=lbl.device)

        return result, gt, n_prompt

    
    def _get_loss(self):
        # # Cross Entropy
        # self.CELoss = nn.CrossEntropyLoss(ignore_index=MASK_CODE)  
        # Dice
        # self.DLoss = DiceLoss(mode='multiclass', 
        #                       from_logits=True, 
        #                       ignore_index=MASK_CODE)
        self.BDLoss = DiceLoss(mode='binary',
                               from_logits=True,
                               ignore_index=MASK_CODE)
        # Focal loss
        # self.FLoss = FocalLoss(mode='multiclass',        # by default, alpha=None, gamma=2.0
        #                        ignore_index=MASK_CODE)
        self.BFLoss = FocalLoss(mode='binary',           
                                ignore_index=MASK_CODE)
        
        # l1 loss
        self.L1Loss = nn.L1Loss()

        return


    def _get_ious(self, masks, y, eps=1e-8):
        ious = []
        for c in range(masks.shape[1]):
            p = (masks[:,c]>=0.5).to(masks.dtype)
            y_ = (y==c).to(masks.dtype)
            tp = (p*y_).sum(axis=(1,2))
            iou = tp/(p.sum(axis=(1,2))+y_.sum(axis=(1,2))-tp+eps)
            ious.append(iou)
        ious = torch.stack(ious, dim=1)
        return ious
    

    def _log_imgs(self, prefix, x, mask, y):
        n_classes = mask.shape[0]
        if n_classes==1:
            mask = (mask[0].detach().cpu().numpy()>0.5).astype(np.uint8)
        else:
            maxp = mask.max(axis=0).values
            mask = mask.argmax(axis=0)+1
            mask[maxp<0.5] = 0
            mask = mask.detach().cpu().numpy().astype(np.uint8)
        class_labels = {i:str(i) for i in range(n_classes+1)}
        y += 1
        y[y>=MASK_CODE] = 0
        masks={'prediction':   {'mask_data': mask, 'class_labels': class_labels},
               'ground truth': {'mask_data': y.detach().cpu().numpy(), 'class_labels': class_labels}
                }
        # get rgb img
        x = F.interpolate(x.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
        if x.shape[0]==3:
            rgb_img = x.detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[0]>3:
            rgb_img = x[0,[3,2,1]].detach().cpu().numpy().transpose((1,2,0))
        mask_img = wandb.Image(rgb_img, masks=masks)
        wandb.log({f'{prefix}_masks': mask_img})


    


