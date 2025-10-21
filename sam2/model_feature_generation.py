import h5py
import random

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
# from PIL.Image import Image
from einops import rearrange

# import lightening
from torch import optim, nn
from torch.nn import functional as F

# import sam2 package
# from sam2.modeling.sam2_base import SAM2Base
# from sam2.utils.transforms import SAM2Transforms
from sam2.build_sam import build_sam2
from sam2.modeling.text_simple_tokenizer import SimpleTokenizer

from dataset.datasets_settings import class_names, class_weight_list   


LOGIT_SCALE = 50
FINAL_LR = 1e-6
KEY_IMG = 'image'
KEY_LBL = 'label'
KEY_DATASAT = 'dataset'
KEY_WAVE = 'waves'
KEY_WEAKLBL = 'weak'
MASK_CODE = 255
INPUT_SIZE = 1024
TEXT_PREFIX = 'A photo of '
CLIP_IMG_SIZE = 224
CLIP_FEAT_SIZE = 224//32


class GenerateFeat():
    def __init__(self, args, use_GPU=True):

    # copnstruct sam2-based model (image encoder and prompt encoder weights are automatically fixed)
        use_dofa = (args.dofa_type not in [None, 'None']) 
        hydra_overrides_extra = [f'++model.image_encoder.trunk.handcrafted_tune={not args.mute_highfreq}',
                                 f'++model.image_encoder.trunk.spectralfeat_tune={use_dofa}',
                                 f'++model.image_encoder.trunk.dofa_name={args.dofa_type}',
                                 f'++model.image_encoder.trunk.dofa_ckpt_path={args.dofa_ckpt_path}',
                                 f'++model.image_encoder.trunk.dofa_wo_rgb={args.dofa_wo_rgb}',
                                 f'++model.use_auxiliary_decoder={args.use_aux_decoder}',
                                 f'++model.decoder_expand_dim={args.de_expand_dim}',
                                 f'++model.decoder_use_opt_two={args.opt_two}',
                                 f'++model.decoder_use_feature_norm={args.use_feature_norm}',
                                 f'++model.clip_use_quick_gelu={args.clip_use_quick_gelu}',]
        if args.mute_adaptor: hydra_overrides_extra += ['++model.image_encoder.trunk.adaptor=None']
        if args.dofa_unfix_weights: 
            hydra_overrides_extra += ['++model.image_encoder.trunk.dofa_unfix_weights=True']
        else:
            assert args.dofa_lr_scale==1, f'dofa_lr_scale={args.dofa_lr_scale}, but dofa_unfix_weights is False!'
        self.dofa_lr_scale = args.dofa_lr_scale
        # logvar smoother
        self.use_logvar_smooth = args.use_logvar_smooth
        if args.use_logvar_smooth:
            hydra_overrides_extra += ['++model.use_logvar_smooth=True']
            if args.fix_logvar_smooth:
                hydra_overrides_extra += ['++model.fix_logvar_smooth=True']
        self.model = build_sam2(args.config_file, args.ts_wpath, args.ckpt_path_clip, 
                                load_decoder_weights=args.load_de,
                                load_aux_decoder_weights=args.load_aux_de,
                                dofa_unfix_weights=args.dofa_unfix_weights, 
                                hydra_overrides_extra=hydra_overrides_extra,
                                de_expand_dim=args.de_expand_dim,
                                test_mode=True)

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        # others TO BE DEFINED
        # self.multimask_output = False
        self.tokenizer = SimpleTokenizer()
        self.share_connect = args.share_connect

        # load weights
        # try:
        #     state_dict = torch.load(args.ts_wpath, map_location='cpu')['state_dict']
        # except:
        #     ckpt = torch.load(args.ts_wpath)
        #     for k in ckpt: print(k)
        #     raise ValueError('Key Error!')
        # self.model.load(state_dict, strict=True)
        self.use_GPU = use_GPU
        if use_GPU:
            self.model.to('cuda')
    
    def predict_with_unique_prompts(self, batch):
        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens, _, _ = self._get_data(batch)
        # bs = x.shape[0]

        # prediction
        with torch.no_grad():
            masks, iou_predictions, precise_predictions, existance_predictions, _, _ = \
                self.forward(x, prompt_tokens, waves, use_aux=False)
        
        if x.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(masks, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return y, y_exist, masks, iou_predictions, precise_predictions, existance_predictions
    

    def predict_with_multiple_prompts(self, batch, return_embeddings=True, imgs=None):
        # get data
        x, y, waves, y_exist, n_prompt, class_id_list, prompt_tokens, dataset_name = self._get_data_with_multiple_prompts(batch)
        num_cls = y_exist.shape[1]
        if imgs is not None: x = imgs

        # prediction
        with torch.no_grad():
            outs = self.forward(x, prompt_tokens, waves, use_aux=False, return_embeddings=return_embeddings)
        masks, iou_predictions, precise_predictions, existance_predictions, img_feats, text_embeddings = outs[:6]
        
        if x.shape[-2:] != masks.shape[-2:]:
            masks_up = F.interpolate(masks, size=x.shape[-2:], mode='bilinear', align_corners=False)
            masks_list = [masks, masks_up]
        else:
            masks_list = [masks]
        
        # # get the largest logits among multiple prompts
        # masks = (masks*LOGIT_SCALE).softmax(1)
        # cls_index = F.one_hot(torch.tensor(class_id_list, device=self.device), num_cls)
        # cls_index = cls_index.view(1, n_prompt, num_cls, 1, 1)
        
        # masks = (masks.unsqueeze(2) * cls_index).max(1)[0]

        out_dict = {'y': y, 'y_exist': y_exist, 'masks_list': masks_list, 
                    'iou_predictions': iou_predictions, 'precise_predictions': precise_predictions, 'existance_predictions': existance_predictions, 
                    'img_feats': img_feats, 'text_embeddings': text_embeddings}

        # return y, y_exist, masks, iou_predictions, precise_predictions, existance_predictions
        if return_embeddings:
            out_dict['output_embeddings'] = outs[-1]
        return out_dict


    def forward(self, x, prompt, waves=None, use_aux=False, return_embeddings=True):  # text prompt: here is the tokenized tensor
        batch_size = x.shape[0]
        # image embeddings
        img_feats = self._get_image_embed(x, waves, use_aux)
        # prompt text embeddings
        n_classes = prompt.shape[0]
        text_embeddings = self._get_text_embeddings(prompt)   # change to CLIP plain output text embeddings
        
        # decoder forward
        outs = self.forward_decoder(img_feats, text_embeddings, n_classes, batch_size, 
                                    use_aux=use_aux, return_embeddings=return_embeddings)
        # parser outputs
        if return_embeddings:
            masks, iou_predictions, precise_predictions, exist_predictions, embeddings = outs
            return masks, iou_predictions, precise_predictions, exist_predictions, img_feats, text_embeddings, embeddings
        else:
            masks, iou_predictions, precise_predictions, exist_predictions = outs        
            return masks, iou_predictions, precise_predictions, exist_predictions, img_feats, text_embeddings
    
    
    def forward_decoder(self, img_feats, text_embeddings, n_classes, batch_size, use_aux=False, return_multiple=False, 
                        one_no_noise_mask_embed=False, return_embeddings=True, 
                        use_mc_dropout = False, p_mc_dropout = 0.5):
        _, _, h, w = img_feats["image_embed"].shape   # [batch_size, 256, 16, 16]
        text_embeddings = self.model.sam_prompt_fn(text_embeddings) # [n_prompt, dim] n_prompt=512, dim=256
        # noise mask embeddings
        if one_no_noise_mask_embed:
            no_noise_mask_embed = self.model.no_noise_mask_embed
        else:
            if type(use_aux)==bool:
                no_noise_mask_embed = self.model.no_noise_mask_embed_aux if use_aux else self.model.no_noise_mask_embed
            else:
                assert type(use_aux)==int, 'use_aux should be either boolean or integer!'
                no_noise_mask_embed = getattr(self.model, f'no_noise_mask_embed_aux_{use_aux}') 
        noise_mask_embeddings = no_noise_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            n_classes*batch_size, -1, h, w
            )
        
        # make predictions
        if type(use_aux)==bool:
            mask_decoder = self.model.sam_mask_decoder_aux if use_aux else self.model.sam_mask_decoder
            high_res_key = 'high_res_feats_aux' if use_aux else 'high_res_feats'
        else:
            mask_decoder = getattr(self.model, f'sam_mask_decoder_aux_{use_aux}')
            high_res_key = f'high_res_feats_aux_{use_aux}'
        outs = self._pred_mask_and_iou(
            mask_decoder,
            img_feats, text_embeddings, noise_mask_embeddings,
            n_classes, batch_size,
            high_res_key,
            return_multiple=return_multiple,
            return_embeddings=return_embeddings,
            use_mc_dropout=use_mc_dropout, 
            p_mc_dropout=p_mc_dropout)
        # parser outputs
        masks, iou_predictions, precise_predictions, exist_predictions = outs[:4]
        
        # parser outputs
        if return_embeddings:
            embeddings = outs[-1]
            return masks, iou_predictions, precise_predictions, exist_predictions, embeddings
        else:
            return masks, iou_predictions, precise_predictions, exist_predictions


    def _get_data(self, batch, return_single=True):
        x = batch[KEY_IMG].contiguous()
        y = batch[KEY_LBL].contiguous()
        if self.use_GPU:
            x = x.to('cuda')
            y = y.to('cuda')

        if return_single:
            assert len(y.shape)==3, f'Label shape is not correct: {y.shape} with return_single={return_single}'
        waves = batch[KEY_WAVE][0]
        if self.use_GPU: waves = waves.to('cuda')
        dataset_name = batch[KEY_DATASAT][0]
        if_weak_lbl = batch[KEY_WEAKLBL][0]
        cls_name_keys = MULTI_MASK_KEYS[dataset_name] if dataset_name in MULTI_MASK_KEYS else [dataset_name]
        ys, y_exists, n_classes, prompt_tokens = [], [], [], []
        for ki, key in enumerate(cls_name_keys):
            cls_all = class_names[key]
            if self.trainer.training:
                CLS_DICT = {i:random.choice(cls_all[i]) for i in cls_all}
            else:
                CLS_DICT = {i:cls_all[i][0] for i in cls_all}
            # existance GT
            y_ = y if len(cls_name_keys)==1 else y[:,ki]
            y_exist = torch.zeros((y_.shape[0], len(CLS_DICT)), device=y_.device)
            for i in range(len(CLS_DICT)):
                ye_ = (y_==i).sum(axis=(1,2))
                y_exist[:,i] = (ye_>0).to(y_exist.dtype)
            
            # prepare prompt
            prompt_tokens_, n_classes_ = self._prepare_prompt(CLS_DICT, y)
            
            # append
            ys.append(y_)
            y_exists.append(y_exist)
            n_classes.append(n_classes_)
            prompt_tokens.append(prompt_tokens_)
        
        if return_single:
            return x, ys[0], waves, y_exists[0], n_classes[0], prompt_tokens[0], if_weak_lbl, cls_name_keys[0]
        else:
            return x, ys, waves, y_exists, n_classes, prompt_tokens, if_weak_lbl, cls_name_keys


    def _get_data_with_multiple_prompts(self, batch):
        x = batch[KEY_IMG].contiguous()
        y = batch[KEY_LBL].contiguous()
        if self.use_GPU:
            x = x.to('cuda')
            y = y.to('cuda')
        
        waves = batch[KEY_WAVE][0]
        if self.use_GPU: waves = waves.to('cuda')
        dataset_name = batch[KEY_DATASAT][0]
        # if_weak_lbl = batch[KEY_WEAKLBL][0]

        # prepare prompts
        if 'nlcd' in dataset_name: 
            dataset_key = 'nlcd-lc'
            y = y[:,0]
        else:
            dataset_key = dataset_name
        cls_all = class_names[dataset_key]
        num_class = len(cls_all)
        CLS_DICT = {}
        class_id_list = []
        idx = 0
        for k in cls_all:
            class_id_list = class_id_list + [k]*len(cls_all[k])
            for cls_str in cls_all[k]:
                CLS_DICT[idx] = cls_str
                idx += 1
        
        # existance GT
        y_exist = torch.zeros((y.shape[0], num_class), device=y.device)
        for i in range(num_class):
            ye_ = (y==i).sum(axis=(1,2))
            y_exist[:,i] = (ye_>0).to(y_exist.dtype)
        
        # prepare prompt
        prompt_tokens, n_prompt = self._prepare_prompt(CLS_DICT, y)
        
        return x, y, waves, y_exist, n_prompt, class_id_list, prompt_tokens, dataset_name


    def _get_image_embed(
        self,
        input_image: Union[torch.Tensor],  # [B, C, H, W]
        wave_list: List[float],
        aux_decoders: List[Union[bool, int]] | bool | int = [],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        if type(aux_decoders) != list:
            aux_decoders = [] if aux_decoders is False else [aux_decoders]
        # assert (
        #     len(input_image.shape) == 4 and input_image.shape[1] == 3
        # ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        # logging.info("Computing image embeddings for the provided image...")
        batch_size = input_image.shape[0]
        # print(f'In trainer-embed: waves - {wave_list}')
        backbone_out = self.model.forward_image(input_image, wave_list)
        # high resolution features for the main decoder
        vision_feats = self.model._prepare_backbone_features(backbone_out["backbone_fpn_de"])
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        # if self.model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}  # self._features

        # high resolution features for the auxiliary decoders
        if len(aux_decoders)>0:
            for aux in aux_decoders:
                assert aux is True or type(aux)==int, 'Auxiliary decoder should be True or an integer!'
                aux_key = f"high_res_feats_aux" if aux is True else f"high_res_feats_aux_{aux}"
                if self.share_connect:
                    features[aux_key] = features["high_res_feats"]
                else:
                    backbone_out_aux = self.model._get_backbone_features_from_aux_decoder(backbone_out, aux)
                    vision_feats_aux = self.model._prepare_backbone_features(backbone_out_aux)
                    vision_feats_aux[-1] = vision_feats_aux[-1] + self.model.no_mem_embed
                    feats_aux = [
                        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                        for feat, feat_size in zip(vision_feats_aux[::-1], self._bb_feat_sizes[::-1])
                    ][::-1]
                    aux_key = f"high_res_feats_aux" if aux is True else f"high_res_feats_aux_{aux}"
                    features[aux_key] = feats_aux[:-1]

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
    

    def _get_text_embeddings(self, prompt):
        text_embeddings = self.model.sam_prompt_encoder(prompt) # [n_prompt, dim] dim=512 (clip dim)
        # text_embeddings = self.model.sam_prompt_fn(text_embeddings) # [n_prompt, dim] dim=256
        return text_embeddings


    def _pred_mask_and_iou(self, decoder, img_feats, text_embeddings, noise_mask_embeddings,
                           n_classes, batch_size, high_res_key='high_res_feats', 
                           return_multiple=False, return_embeddings=False,
                           use_mc_dropout=False, p_mc_dropout=0.5):
        # rearrange text embeddings
        text_embeddings = rearrange(text_embeddings.unsqueeze(0).expand(batch_size, -1, -1),
                                    'batch_size n_classes dim -> (n_classes batch_size) dim', 
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        
        # get high resolution features
        high_res_features = [
            feat_level.repeat(n_classes, 1, 1, 1) for feat_level in img_feats[high_res_key]
        ]

        # get predictions
        outs = decoder(
            image_embeddings=img_feats["image_embed"].repeat(n_classes, 1, 1, 1),
            text_embeddings=text_embeddings.unsqueeze(1),
            noise_mask_embeddings=noise_mask_embeddings,
            high_res_features=high_res_features,
            multimask_output=return_multiple,
            return_embeddings=return_embeddings,
            use_mc_dropout=use_mc_dropout,
            p_mc_dropout=p_mc_dropout,
        )
        masks, iou_predictions, precise_predictions, exist_predictions = outs[:4]

        # mask predictions
        # masks = (torch.clamp(masks, -32.0, 32.0) + 32.0)/64.0
        if return_multiple:
            n_mask = masks.shape[1]
            masks = rearrange(masks,
                              '(n_classes batch_size) n_mask H W -> n_mask batch_size n_classes H W',
                              batch_size=batch_size,
                              n_classes=n_classes,
                              n_mask=n_mask)
        else:
            masks = torch.clamp(masks, -32.0, 32.0)
            masks = rearrange(torch.squeeze(masks, dim=1), 
                            '(n_classes batch_size) H W -> batch_size n_classes H W',
                            batch_size=batch_size,
                            n_classes=n_classes)
            
        # accuracy predictions
        iou_predictions = rearrange(iou_predictions[:,0], 
                                    '(n_classes batch_size) -> batch_size n_classes',
                                    batch_size=batch_size,
                                    n_classes=n_classes)
        precise_predictions = rearrange(precise_predictions[:,0], 
                                        '(n_classes batch_size) -> batch_size n_classes',
                                        batch_size=batch_size,
                                        n_classes=n_classes)
        exist_predictions = torch.clamp(exist_predictions, -32.0, 32.0)
        exist_predictions = rearrange(exist_predictions[:,0],
                                      '(n_classes batch_size) -> batch_size n_classes',
                                      batch_size=batch_size,
                                      n_classes=n_classes)
        
        # outputs
        if return_embeddings:
            embeddings = outs[-1]
            return masks, iou_predictions, precise_predictions, exist_predictions, embeddings
        else:
            return masks, iou_predictions, precise_predictions, exist_predictions
    

    
    
