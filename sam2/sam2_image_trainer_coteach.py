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
        self.model = build_sam2(args.config_file, args.ckpt_path, args.ckpt_path_clip, args.load_de)
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
        self.n_train_batches = int(args.n_batch_tr/(args.nodes*args.devices))
        # display interval
        self.display_inv = args.display_inv
        
        # co-teaching
        assert self.model.use_auxiliary_decoder, "Auxiliary decoder is required for using weak labels with co-teaching"
        # for sample selection
        self.rampup_type = args.rampup_type
        self.rampup_length = args.rampup_length
        self.init_forget_rate = args.init_forget_rate  # initial forget rate
        self.forget_rate = args.forget_rate            # final forget rate
        assert self.init_forget_rate<=self.forget_rate, \
                f'Initial forget rate ({self.init_forget_rate}) should be less than the final one ({self.forget_rate})!'
        
        # # This property activates manual optimization.
        # self.automatic_optimization = False


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
        high_res_features_aux = [
            feat_level.repeat(n_classes, 1, 1, 1) for feat_level in img_feats["high_res_feats_aux"]
        ]
        # decoder
        masks, iou_predictions = self._pred_mask_and_iou(
            self.model.sam_mask_decoder,
            img_feats["image_embed"], text_embeddings, high_res_features, n_classes, batch_size)
        # auxiliary decoder
        masks_aux, iou_predictions_aux = self._pred_mask_and_iou(
            self.model.sam_mask_decoder_aux,
            img_feats["image_embed"], text_embeddings, high_res_features_aux, n_classes, batch_size)
        
        return masks, masks_aux, iou_predictions, iou_predictions_aux
    

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
                interval = 'epoch'
            elif self.schdt=='cos':   # [T_warm, T, lr_min] warm-up length, total length of cos scheduler, final lr
                T_warm = int(self.schdv[0])
                T = int(self.schdv[1])
                if len(self.schdv)>=3:
                    lr_min = self.schdv[2]
                    if len(self.schdv)>3:
                        print(f'More values than three ({self.schdv}) are provided for cosine annealing scheduler, only the first three ({T_warm, T, lr_min}) are used!')
                else:
                    lr_min = FINAL_LR
                print(f'Initial LR (={self.lr}) is scheduled by {self.schdt} to reduce to {lr_min} in {T} epochs with a {T_warm}-epoch linear warmup stage.')
                # lambda_func
                step_warmup = T_warm*self.n_train_batches
                step_total = T*self.n_train_batches
                lambda_func = lambda step: (
                    # Warmup phase: Linearly increase learning rate from warmup_lr to final_lr
                    step / step_warmup
                    if step < step_warmup
                    else (
                    # Cosine annealing phase after warmup
                    (lr_min + (1 + np.cos(np.pi * (step - step_warmup) / (step_total - step_warmup))) * (self.lr - lr_min) / 2) / self.lr
                    if step < step_total 
                    else 
                    lr_min / self.lr)
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
                interval = 'step'
            elif self.schdt=='exp':
                gamma = self.schdv[0]
                if len(self.schdv)>1:
                    print(f'More values than one ({self.schdv}) are provided for exponential scheduler, only the first one ({gamma}) is used!')
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                interval = 'epoch'
                print(f'Initial LR (={self.lr}) is scheduled by {self.schdt} with gamma={gamma}.')
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval},}
    

    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        # main step
        # # optimizers
        # opt = self.optimizers()
        # if self.schdt!='none': sch, sch_aux = self.lr_schedulers()

        # get data
        x = batch[KEY_IMG]
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).contiguous()
        y = batch[KEY_LBL]
        prompt_tokens, y, n_classes = self._prepare_prompt(CLS_DICT, y)
        # lbl_type = batch[KEY_LBLTYPE]  # indicate the type of labels: 0 - gt, 1 - ns
        # assert lbl_type in [0, 1], 'Please provide correct label type (0 - gt, 1 - ns)!'
        
        # prediction
        masks, masks_aux, iou_predictions, iou_predictions_aux = self.forward(x, prompt_tokens)

        # sample selection
        m_discard, m_discard_aux = self._get_selection_masks(masks.detach(), 
                                                             masks_aux.detach(), 
                                                             y.detach(), n_classes)
        y_org, y_aux = y.clone(), y.clone()
        y[m_discard] = MASK_CODE
        y_aux[m_discard_aux] = MASK_CODE

        # calculate losses
        loss, loss_aux = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)       
        # loss
        for L, w in zip([self.BDLoss, self.BFLoss], [self.wdice, self.wfocal]):
            for c in range(n_classes):
                # loss  
                yc = (y==c).to(y.dtype)
                yc[y==MASK_CODE] = MASK_CODE
                loss_ = w*L(masks[:,c], yc)
                loss += loss_
                # loss_aux
                yc = (y_aux==c).to(y_aux.dtype)
                yc[y_aux==MASK_CODE] = MASK_CODE
                loss_aux_ = w*L(masks_aux[:,c], yc)
                loss_aux += loss_aux_
        self.log('tr_loss_mask', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('tr_loss_mask_aux', loss_aux, on_epoch=True, on_step=False, sync_dist=True)
        
        # iou loss
        ious = self._get_ious(masks.detach(), y_org)
        ious_aux = self._get_ious(masks_aux.detach(), y_org)
        loss += self.L1Loss(iou_predictions, ious)*n_classes
        loss_aux += self.L1Loss(iou_predictions_aux, ious_aux)*n_classes

        self.log('tr_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('tr_loss_aux', loss_aux, on_epoch=True, on_step=False, sync_dist=True)
        
        # training image examples  
        if batch_idx==0 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
            self._log_imgs('tr', x[0], masks[0], y_org[0])
            self._log_imgs('tr_aux', x[0], masks_aux[0], y_aux[0])
        
        # opt.zero_grad()
        # self.manual_backward(loss+loss_aux)
        # opt.step()

        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(self.device, name)
        return loss+loss_aux
    

    def validation_step(self, batch, batch_idx):
        # get data
        x = batch[KEY_IMG]
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).contiguous()
        y = batch[KEY_LBL]
        # prompt = batch[KEY_PROMPT]
        prompt_tokens, y, n_classes = self._prepare_prompt(CLS_DICT, y, train_mode=False)
        
        # prediction
        masks, masks_aux, iou_predictions, iou_predictions_aux = self.forward(x, prompt_tokens)
        
        # ious
        ious = self._get_ious(masks.detach(), y)
        self.log('val_mious', ious.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mious_pred', iou_predictions.mean(), on_epoch=True, on_step=False, sync_dist=True)
        if batch_idx==0 and self.trainer.global_rank==0:
            for i in range(4):
                self._log_imgs('val', x[i], masks[i], y[i])
        # iou auxilary
        ious_aux = self._get_ious(masks_aux.detach(), y)
        self.log('val_mious_aux', ious_aux.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('val_mious_pred_aux', iou_predictions_aux.mean(), on_epoch=True, on_step=False, sync_dist=True)
        if batch_idx==0 and self.trainer.global_rank==0:
            for i in range(4):
                self._log_imgs('val_aux', x[i], masks_aux[i], y[i])


    def test_step(self, batch, batch_idx):
        # get data
        x = batch[KEY_IMG]
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False).contiguous()
        y = batch[KEY_LBL]
        # prompt = batch[KEY_PROMPT]
        prompt_tokens, y, n_classes = self._prepare_prompt(CLS_DICT, y, train_mode=False)
        
        # prediction
        masks, masks_aux, iou_predictions, iou_predictions_aux = self.forward(x, prompt_tokens)

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
        vision_feats, vision_feats_aux = self.model._prepare_backbone_features(backbone_out)
        # print("vision_feats:", len(vision_feats))
        # for i in vision_feats: print(self.device, i.shape)
        # print("vision_feats_aux:", len(vision_feats_aux))
        # for i in vision_feats_aux: print(self.device, 'aux', i.shape)
        # _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        # if self.model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        feats_aux = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats_aux[::-1], self._bb_feat_sizes[:-1][::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1], "high_res_feats_aux": feats_aux}
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


    def _pred_mask_and_iou(self, decoder, img_embeddings, text_embeddings, high_res_features, n_classes, batch_size):
        masks, iou_predictions = decoder(
            image_embeddings=img_embeddings.repeat(n_classes, 1, 1, 1),
            multimask_output=self.multimask_output,
            text_embeddings=text_embeddings.unsqueeze(1),
            high_res_features=high_res_features,
        )

        # masks = (torch.clamp(masks, -32.0, 32.0) + 32.0)/64.0
        masks = torch.clamp(masks, -32.0, 32.0)
        masks = rearrange(torch.squeeze(masks, dim=1), 
                          '(n_classes batch_size) H W -> batch_size n_classes H W',
                          batch_size=batch_size,
                          n_classes=n_classes)
        iou_predictions = rearrange(torch.squeeze(iou_predictions, dim=1), 
                                    '(n_classes batch_size) -> batch_size n_classes',
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        return masks, iou_predictions

    
    def _get_loss(self):
        # Cross Entropy - for sample selection
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')  
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
    

    def _get_selection_masks(self, p, p_aux, y, n_classes):
        for v in [p, p_aux, y]:
            if v.grad_fn is not None:
                raise ValueError('The input tensors for the selection mask generation fucntion should not require gradients!')

        # get number of samples to select
        forget_rate = self._get_current_forget_rate(self.current_epoch)
        remember_rate = 1 - forget_rate
        self.log('forget_rate', forget_rate, on_epoch=True, on_step=False, sync_dist=True)
        
        # initialize discard masks
        m_discard = torch.zeros_like(y, dtype=torch.bool, device=self.device, requires_grad=False)
        m_discard_aux = torch.zeros_like(y, dtype=torch.bool, device=self.device, requires_grad=False)
        
        if forget_rate>0:
            # calculate BCE losses
            l_select = torch.zeros_like(y, dtype=p.dtype, device=self.device, requires_grad=False)
            l_select_aux = torch.zeros_like(y, dtype=p.dtype, device=self.device, requires_grad=False)
            for c in range(n_classes):
                y_ = (y==c).float()
                l_select += self.BCELoss(p[:,c], y_)
                l_select_aux += self.BCELoss(p_aux[:,c], y_)
            
            # select samples
            for c in range(n_classes):
                if (y==c).sum()>0:
                    thr = torch.quantile(l_select[y==c].float(), remember_rate)
                    thr_aux = torch.quantile(l_select_aux[y==c].float(), remember_rate)
                    m_discard[y==c] = (l_select_aux[y==c]>thr_aux)
                    m_discard_aux[y==c] = (l_select[y==c]>thr)
        return m_discard, m_discard_aux


    def _get_ious(self, masks, y, eps=1e-8):
        ious = []
        for c in range(masks.shape[1]):
            p = (masks[:,c]>=0.5).to(masks.dtype)
            y_ = (y==c).to(masks.dtype)
            tp = (p*y_).sum(axis=(1,2))
            iou = (tp+eps)/(p.sum(axis=(1,2))+y_.sum(axis=(1,2))-tp+eps)
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


    def _get_current_forget_rate(self, current_epoch):
        exp_ind = 5
        cepoch = min(current_epoch, self.rampup_length)
        if self.rampup_type=='linear':
            current_forget_rate = self.init_forget_rate + \
                                    (self.forget_rate-self.init_forget_rate)*cepoch/self.rampup_length
        elif self.rampup_type=='sigmoid':
            """sigmoid rampup from https://arxiv.org/abs/1610.02242"""
            # assert exp_ind>=5, 'sigmoid index should be greater than 5!'
            if current_epoch==0:
                current_forget_rate = 0.
            else:
                x = 1-cepoch/self.rampup_length
                current_forget_rate = self.init_forget_rate + \
                                        np.exp(-exp_ind * (x**2)) * (self.forget_rate-self.init_forget_rate)
        else:
            current_forget_rate = self.forget_rate
        return current_forget_rate
    


