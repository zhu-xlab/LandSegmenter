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

from dataset.datasets_settings import class_names

FINAL_LR = 1e-6
KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
MASK_CODE = 255
INPUT_SIZE = 1024
TEXT_PREFIX = 'A photo of '

# CLS_DICT = {0:'Water', 1:'Trees', 2:'Grass', 3:'Flooded vegetation', 4:'Crops', 5:'Shrub and Scrub', 6:'Built area', 7:'Bare land', 8:'Ice and Snow'}


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
        use_dofa = (args.dofa_type not in [None, 'None']) 
        hydra_overrides_extra = [f'++model.image_encoder.trunk.spectralfeat_tune={use_dofa}',
                                 f'++model.image_encoder.trunk.dofa_name={args.dofa_type}',
                                 f'++model.image_encoder.trunk.dofa_ckpt_path={args.dofa_ckpt_path}',
                                 f'++model.use_auxiliary_decoder={args.use_aux_decoder}',]
        if args.mute_adaptor: hydra_overrides_extra += ['++model.image_encoder.trunk.adaptor=None']
        if args.dofa_unfix_weights: hydra_overrides_extra += ['++model.image_encoder.trunk.dofa_unfix_weights=True']
        self.model = build_sam2(args.config_file, args.ckpt_path, args.ckpt_path_clip, 
                                load_decoder_weights=args.load_de,
                                load_aux_decoder_weights=args.load_aux_de,
                                dofa_unfix_weights=args.dofa_unfix_weights, 
                                hydra_overrides_extra=hydra_overrides_extra)

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        # others TO BE DEFINED
        self.multimask_output = False
        self.tokenizer = SimpleTokenizer()
        # self.n_prompts = args.n_prompts if type(args.n_prompts)==type([]) else [args.n_prompts]
        # loss weights
        self.wfocal = args.wfocal
        self.wdice = args.wdice
        self.wiou_no = args.wiou_no
        # set up the losses: dice and focal
        self.use_multiclass_loss = args.use_multiclass_loss
        self._set_loss_functions()
        # self.wunsure = args.winlbl  # incompletely labeled patches are regarded as unsure, assigning a smaller weight
        # optimizer
        self.opt = args.opt
        self.lr = args.lr
        # scheduler
        self.schdt = args.schdt
        self.schdv = args.schdv
        self.max_epochs = args.epochs
        self.n_train_batches = int(args.n_batch_tr/args.ntasks)
        # display interval
        self.display_inv = args.display_inv

    # ------ forward propogation ------ #
    def forward(self, x, prompt, waves=None, use_aux=False):  # text prompt: here is the tokenized tensor
        batch_size = x.shape[0]
        # image embeddings
        # print(f'In trainer-forward: waves - {waves}')
        img_feats = self._get_image_embed(x, waves)
        # prompt text embeddings
        n_classes = prompt.shape[0]
        text_embeddings = self.model.sam_prompt_encoder(prompt) # [n_prompt, dim] dim=512 (clip dim)
        text_embeddings = self.model.sam_prompt_fn(text_embeddings) # [n_prompt, dim] dim=256
        
        # decoder forward
        masks, iou_predictions, precise_predictions, exist_predictions = \
            self.forward_decoder(img_feats, text_embeddings, n_classes, batch_size, use_aux=use_aux)
        
        return masks, iou_predictions, precise_predictions, exist_predictions, img_feats, text_embeddings
    

    def forward_decoder(self, img_feats, text_embeddings, n_classes, batch_size, use_aux=False):
        _, _, h, w = img_feats["image_embed"].shape   # [batch_size, 256, 16, 16]
        # noise mask embeddings
        no_noise_mask_embed = self.model.no_noise_mask_embed_aux if use_aux else self.model.no_noise_mask_embed
        noise_mask_embeddings = no_noise_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            n_classes*batch_size, -1, h, w
            )
        
        # make predictions
        mask_decoder = self.model.sam_mask_decoder_aux if use_aux else self.model.sam_mask_decoder
        masks, iou_predictions, precise_predictions, exist_predictions = self._pred_mask_and_iou(
            mask_decoder,
            img_feats, text_embeddings, noise_mask_embeddings,
            n_classes, batch_size)
        
        return masks, iou_predictions, precise_predictions, exist_predictions

    
    def forward_decoder_w_mask(self, noise_mask, img_feats, text_embeddings, use_aux=False):
        n_classes = text_embeddings.shape[0]
        batch_size = noise_mask.shape[0]
        # prepare binary masks
        if noise_mask.dim()==3:
            mask_w_nodata = torch.sum(noise_mask==MASK_CODE)>0
            if mask_w_nodata:
                nodata_mask = noise_mask==MASK_CODE
                noise_mask[nodata_mask] = 0
            noise_mask = F.one_hot(noise_mask, n_classes).permute(0,3,1,2).float()
            if mask_w_nodata:
                noise_mask[:,0][nodata_mask] = 0
        # rearrange noise mask to be compatible with image feature embeddings
        noise_mask = rearrange(noise_mask.to(text_embeddings.dtype), 'b c h w -> (c b) 1 h w')
        # noise mask embeddings
        sam_noise_mask_downscaling = self.model.sam_noise_mask_downscaling_aux if use_aux else self.model.sam_noise_mask_downscaling
        noise_mask_embeddings = sam_noise_mask_downscaling(noise_mask)

        # make predictions
        mask_decoder = self.model.sam_mask_decoder_aux if use_aux else self.model.sam_mask_decoder
        masks, iou_predictions, precise_predictions, exist_predictions = self._pred_mask_and_iou(
            mask_decoder,
            img_feats, text_embeddings, noise_mask_embeddings,
            n_classes, batch_size)
        
        return masks, iou_predictions, precise_predictions, exist_predictions
        

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
            elif self.schdt=='cos':   # [T_warm, T, lr_min] warm-up length, total length of cos scheduler, final lr
                T_warm = int(self.schdv[0])
                T = int(self.schdv[1])
                if len(self.schdv)>=3:
                    lr_min = self.schdv[2]
                    if len(self.schdv)>3:
                        print(f'More values than two ({self.schdv}) are provided for cosine annealing scheduler, only the first two ({T_warm, T, lr_min}) are used!')
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
            elif self.schdt=='exp':
                gamma = self.schdv[0]
                if len(self.schdv)>1:
                    print(f'More values than one ({self.schdv}) are provided for exponential scheduler, only the first one ({gamma}) is used!')
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                print(f'Initial LR (={self.lr}) is scheduled by {self.schdt} with gamma={gamma}.')
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"},}
    

    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        # main step
        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens = self._get_data(batch)
        
        # prediction
        masks, iou_predictions, precise_predictions, exist_predictions, _, _ = \
            self.forward(x, prompt_tokens, waves)
        
        # prediction accuracies
        precises, ious = self._get_accs(masks.detach(), y)
        
        # calculate losses 
        loss = self._calculate_loss(n_classes, y, y_exist, ious, precises, 
                        masks, exist_predictions, iou_predictions, precise_predictions,
                        )
        
        # training image examples  
        if batch_idx==0 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
            self._log_imgs('tr', x[0], masks[0], y[0])
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        self._do_validation(batch, batch_idx, prefix='val')


    def test_step(self, batch, batch_idx):
        self._do_validation(batch, batch_idx, prefix='test')


    # ------ Model utilities ------ #
    def _get_data(self, batch):
        x = batch[KEY_IMG].contiguous()
        y = batch[KEY_LBL].contiguous()
        waves = batch[KEY_WAVE][0]
        dataset_name = batch[KEY_DATASAT][0]
        cls_all = class_names[dataset_name]
        if self.trainer.training:
            CLS_DICT = {i:random.choice(cls_all[i]) for i in cls_all}
        else:
            CLS_DICT = {i:cls_all[i][0] for i in cls_all}
        # existance GT
        y_exist = torch.zeros((y.shape[0], len(CLS_DICT)), device=self.device)
        for i in range(len(CLS_DICT)):
            ye_ = (y==i).sum(axis=(1,2))
            y_exist[:,i] = (ye_>0).to(y_exist.dtype)
        
        # prepare prompt
        prompt_tokens, n_classes = self._prepare_prompt(CLS_DICT, y)
        # lbl_type = batch[KEY_LBLTYPE]  # indicate the type of labels: 0 - gt, 1 - ns
        # assert lbl_type in [0, 1], 'Please provide correct label type (0 - gt, 1 - ns)!'

        return x, y, waves, y_exist, n_classes, prompt_tokens


    def _get_image_embed(
        self,
        input_image: Union[torch.Tensor],  # [B, C, H, W]
        wave_list: List[float],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        # assert (
        #     len(input_image.shape) == 4 and input_image.shape[1] == 3
        # ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        batch_size = input_image.shape[0]
        # print(f'In trainer-embed: waves - {wave_list}')
        backbone_out = self.model.forward_image(input_image, wave_list)
        vision_feats = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        # if self.model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}  # self._features
        return features
    

    def _prepare_prompt(self, class_name_dict, lbl):
        """ Tokenization of string inputs for the model. """
        truncate = False
        context_length = self.model.sam_prompt_encoder.context_length

        # generate prompt texts and reprepare labels
        nc_total = len(class_name_dict)
        n_prompt = nc_total
        texts = [TEXT_PREFIX+class_name_dict[i] for i in range(nc_total)]

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

        return result, n_prompt


    def _pred_mask_and_iou(self, decoder, img_feats, text_embeddings, noise_mask_embeddings,
                           n_classes, batch_size):
        # rearrange text embeddings
        text_embeddings = rearrange(text_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
                                    'batch_size n_classes dim -> (n_classes batch_size) dim', 
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        
        # get high resolution features
        high_res_features = [
            feat_level.repeat(n_classes, 1, 1, 1) for feat_level in img_feats["high_res_feats"]
        ]

        # get predictions
        masks, iou_predictions, precise_predictions, exist_predictions = decoder(
            image_embeddings=img_feats["image_embed"].repeat(n_classes, 1, 1, 1),
            multimask_output=self.multimask_output,
            text_embeddings=text_embeddings.unsqueeze(1),
            noise_mask_embeddings=noise_mask_embeddings,
            high_res_features=high_res_features,
        )

        # masks = (torch.clamp(masks, -32.0, 32.0) + 32.0)/64.0
        masks = torch.clamp(masks, -32.0, 32.0)
        masks = rearrange(torch.squeeze(masks, dim=1), 
                          '(n_classes batch_size) H W -> batch_size n_classes H W',
                          batch_size=batch_size,
                          n_classes=n_classes)
        
        # accuracy predictions
        iou_predictions = rearrange(torch.squeeze(iou_predictions, dim=1), 
                                    '(n_classes batch_size) -> batch_size n_classes',
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        precise_predictions = rearrange(torch.squeeze(precise_predictions, dim=1), 
                                        '(n_classes batch_size) -> batch_size n_classes',
                                        batch_size=batch_size,
                                        n_classes=n_classes)
        exist_predictions = torch.clamp(exist_predictions, -32.0, 32.0)
        exist_predictions = rearrange(torch.squeeze(exist_predictions, dim=1),
                                      '(n_classes batch_size) -> batch_size n_classes',
                                      batch_size=batch_size,
                                      n_classes=n_classes)
        
        # outputs
        return masks, iou_predictions, precise_predictions, exist_predictions
    
    
    def _set_loss_functions(self):
        # Cross Entropy
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')  
        # Dice
        self.BDLoss = DiceLoss(mode='binary',
                               from_logits=True,
                               ignore_index=MASK_CODE)
        # Focal loss
        self.BFLoss = FocalLoss(mode='binary',           
                                ignore_index=MASK_CODE)
        
        # l1 loss
        self.L1Loss = nn.L1Loss(reduction='none')

        # multi-class loss
        if self.use_multiclass_loss:
            self.DLoss = DiceLoss(mode='multiclass', 
                                  from_logits=True, 
                                  ignore_index=MASK_CODE)
            self.FLoss = FocalLoss(mode='multiclass',        # by default, alpha=None, gamma=2.0
                                   ignore_index=MASK_CODE)
            
        return


    def _get_accs(self, masks, y, eps=1e-8):
        ious, precises = [], []
        for c in range(masks.shape[1]):
            p = (masks[:,c]>=0).to(masks.dtype)  # mask here is logits in [-32, 32]
            p[p==MASK_CODE] = 0
            y_ = (y==c).to(masks.dtype)
            tp = (p*y_).sum(axis=(1,2))
            precise = (tp+eps)/(p.sum(axis=(1,2))+eps)
            iou = (tp+eps)/(p.sum(axis=(1,2))+y_.sum(axis=(1,2))-tp+eps)
            precises.append(precise)
            ious.append(iou)
        precises = torch.stack(precises, dim=1)
        ious = torch.stack(ious, dim=1)
        return precises, ious
    

    def _get_exist_accuracy(self, exist_pred, y_exist):
        exist_pred = (exist_pred>0).to(y_exist.dtype)
        acc = (exist_pred==y_exist).sum()/y_exist.numel()
        return acc
    

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
        # x = F.interpolate(x.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
        if x.shape[0]==3:
            rgb_img = x.detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[0]>3:
            rgb_img = x[:3].detach().cpu().numpy().transpose((1,2,0))
        mask_img = wandb.Image(rgb_img, masks=masks)
        wandb.log({f'{prefix}_masks': mask_img})


    def _do_validation(self, batch, batch_idx, prefix, use_aux=False):
        aux_tag = '_aux' if use_aux else ''

        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens = self._get_data(batch)

        # prediction
        masks, iou_predictions, precise_predictions, existance_predictions, _, _ = \
            self.forward(x, prompt_tokens, waves, use_aux=use_aux)
        
        # ious
        _, ious = self._get_accs(masks.detach(), y)
        exist_acc = self._get_exist_accuracy(existance_predictions.detach(), y_exist)
        y_exist = y_exist.to(torch.bool)
        self.log(f'{prefix}{aux_tag}_mious_exist', ious[y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{prefix}{aux_tag}_mious_noexist', ious[~y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{prefix}{aux_tag}_mious_pred_exist', iou_predictions[y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{prefix}{aux_tag}_mious_pred_noexist', iou_predictions[~y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{prefix}{aux_tag}_exist_acc', exist_acc, on_epoch=True, on_step=False, sync_dist=True)
        if batch_idx==0 and self.trainer.global_rank==0:
            for i in range(1):
                self._log_imgs(prefix+aux_tag, x[i], masks[i], y[i])
    

    def _calculate_loss(self, n_classes, y, y_exist, ious, precises,
                        mask_predictions, exist_predictions, iou_predictions, precise_predictions,
                        from_aux_decoder=False):
        aux_tag = '_aux' if from_aux_decoder else ''
        # calculate losses 
        loss = torch.tensor(0.0, device=self.device)
        # binary loss
        for L, w in zip([self.BDLoss, self.BFLoss], [self.wdice, self.wfocal]):
            for c in range(n_classes):  
                y_ = (y==c).to(y.dtype)
                y_[y==MASK_CODE] = MASK_CODE
                loss_ = w*L(mask_predictions[:,c], y_)
                loss += loss_
        # multi-class loss
        if self.use_multiclass_loss:
            for L, w in zip([self.DLoss, self.FLoss], [self.wdice, self.wfocal]):
                loss_ = w*L(mask_predictions, y)
                loss += loss_
        self.log(f'tr{aux_tag}_loss_mask', loss, on_epoch=True, on_step=False, sync_dist=True)
        
        # existance loss
        loss_exist_matrix = self.BCELoss(exist_predictions, y_exist)
        # calculate the proportion of labeled pixels to serve as the weights
        y_proportions = ((y!=MASK_CODE).sum(axis=(1,2))/(y.shape[1]*y.shape[2]))[:,None]
        y_proportions = y_proportions.repeat(1, n_classes)
        w_exist = torch.max(y_proportions, y_exist)
        # weighted existance loss according to the reliability
        loss_exist = (w_exist*loss_exist_matrix).sum()/loss_exist_matrix.numel()
        loss_exist = loss_exist * n_classes
        self.log(f'tr{aux_tag}_loss_exist', loss_exist, on_epoch=True, on_step=False, sync_dist=True)

        # iou loss
        loss_iou_matrix = self.L1Loss(iou_predictions, ious)
        loss_precise_matrix = self.L1Loss(precise_predictions, precises)
        # calculate weighted iou loss
        # weighting strategy: considering 1. the reliability of the existance prediction, 2. the reliability of the iou prediction
        loss_iou = self._calculate_acc_loss(loss_iou_matrix, y_proportions, y_exist)
        loss_precise = self._calculate_acc_loss(loss_precise_matrix, y_proportions, y_exist)
        
        # final loss
        loss = loss \
            + loss_iou*n_classes \
            + loss_precise*n_classes \
            + loss_exist
        self.log(f'tr{aux_tag}_loss', loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    
    def _calculate_acc_loss(self, acc_loss_matrix, y_proportions, y_exist):
        loss_acc = ((y_proportions[y_exist>0] * acc_loss_matrix[y_exist>0]).sum() \
                    + self.wiou_no * (y_proportions[y_exist==0] * acc_loss_matrix[y_exist==0]).sum()\
                    ) / acc_loss_matrix.numel()
        return loss_acc
        


