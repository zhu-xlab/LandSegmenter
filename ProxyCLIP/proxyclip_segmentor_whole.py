import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import openai_imagenet_template, sub_imagenet_template

# from mmseg.models.segmentors import BaseSegmentor
# from mmseg.models.data_preprocessor import SegDataPreProcessor
# from mmengine.structures import PixelData

# from mmseg.registry import MODELS

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

from open_clip import create_model, tokenizer
from segment_anything import sam_model_registry

from mae import models_vit
from mae.pos_embed import interpolate_pos_embed

from myutils import UnNormalize

from sam2.build_sam import build_sam2
from sam2.model_feature_generation import GenerateFeat
from dataset.datasets_settings import class_names


# @MODELS.register_module()
class ProxyCLIPSegmentation():
    def __init__(self, clip_type, model_type, vfm_model, name_path, checkpoint=None, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336, args=None):

        # data_preprocessor = SegDataPreProcessor(
        #     mean=[122.771, 116.746, 104.094],
        #     std=[68.501, 66.632, 70.323],
        #     bgr_to_rgb=True
        # )
        super().__init__() # data_preprocessor=data_preprocessor)

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = vfm_model
        if vfm_model == 'sam':
            self.vfm = sam_model_registry["vit_b"](checkpoint=checkpoint)
            # self.vfm = sam_model_registry["vit_l"](checkpoint=checkpoint)
        
        elif vfm_model == 'sam2':
            hydra_overrides_extra = ['++model.image_encoder.trunk.adaptor=None']
            self.vfm = build_sam2(args.config_file, args.ckpt_path,
                                  hydra_overrides_extra=hydra_overrides_extra).image_encoder
        
        elif vfm_model == 'weak':
            assert args is not None, 'args should be provided for weakly supervised model'
            self.vfm = GenerateFeat(args)
            self.weak_feat_type = args.weak_feat_type

        elif vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif vfm_model == 'dinov2':
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        elif vfm_model == 'mae':
            self.vfm = models_vit.__dict__['vit_base_patch16'](img_size=slide_crop, num_classes=0, global_pool=False)
            checkpoint_model = torch.load(checkpoint, map_location='cpu')['model']
            state_dict = self.vfm.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(self.vfm, checkpoint_model)
            # load pre-trained model
            self.vfm.load_state_dict(checkpoint_model, strict=False)
        else:
            print("vlm_model not supported")

        if vfm_model == 'weak':
            # self.vfm.model = self.vfm.model.half()
            for p in self.vfm.model.parameters():
                p.requires_grad = False
        else:
            # self.vfm = self.vfm.half()
            for p in self.vfm.parameters():
                p.requires_grad = False
            self.vfm.eval().to(device)

        # self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        # self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

        if vfm_model == 'weak':
            cls_dict = class_names[args.dataset_name]
            if self.weak_feat_type != 'image_embed':
                n_per_cls = [len(cls_dict[k]) for k in cls_dict]
                # accum_n_per_cls = [0] + [sum(n_per_cls[:i+1]) for i in range(len(n_per_cls))]
                self.cls_split = {}
                for i in range(len(self.query_idx)):
                    cq = self.query_idx[i]
                    self.cls_split[i] = list(range(sum(n_per_cls[:cq]),sum(n_per_cls[:cq+1])))
                # print(args.dataset_name, self.cls_split)

            self.class_id_list = []
            for k in cls_dict:
                self.class_id_list = self.class_id_list + [k]*len(cls_dict[k])
            self.class_id_list = torch.Tensor(self.class_id_list).to(torch.int64).to(device)

    @torch.no_grad()
    def forward_feature(self, img, ex_feats, logit_size=None):
        if type(img) == list:
            img = img[0]

        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1]

        # imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        # imgs_norm = torch.stack(imgs_norm, dim=0)

        # imgs_norm = imgs_norm.half()

        imgs_norm = img.half()
            
        I, J = ex_feats.shape[-2:]

        image_features = self.clip.encode_image(img.half(),
                                               external_feats=ex_feats,
                                               beta=self.beta,
                                               gamma=self.gamma)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        if self.vfm_model == 'weak' and self.weak_feat_type == 'preout_embed':
            image_features = rearrange(image_features, '(c n) l d -> c n l d', n=img.shape[0])
            logits = []
            for c in range(len(self.query_idx)):
                logits_ = image_features[self.cls_split[c]] @ self.query_features[[c]].T
                logits.append(logits_.mean(dim=0))
            logits = torch.concatenate(logits, dim=-1)
        else:
            logits = image_features @ self.query_features.T
        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits #, mid_feats

    def forward_slide(self, img, img_metas, embed, embed_crop_size, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # crop embeddings
                x1_embed = int(x1//embed_crop_size)
                x2_embed = int(x2//embed_crop_size)
                y1_embed = int(y1//embed_crop_size)
                y2_embed = int(y2//embed_crop_size)
                crop_embed = embed[:, :, y1_embed:y2_embed, x1_embed:x2_embed]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    raise ValueError(f'Please provde crop size of 56*n. Current: {crop_img.shape[2:]}')
                    # crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                crop_seg_logit = self.forward_feature(crop_img, crop_embed).detach()

                torch.cuda.empty_cache()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    def predict(self, inputs, batch=None, return_prob=False):
        batch_img_metas = [
                                dict(
                                    ori_shape=inputs.shape[2:],
                                    img_shape=inputs.shape[2:],
                                    pad_shape=inputs.shape[2:],
                                    padding_size=[0, 0, 0, 0])
                            ] * inputs.shape[0]
        with autocast(): 
            weak_pred, weak_logits, embed = self.get_weak_model_prediction(batch, return_embeddings=True)
            weak_patch_size = inputs.shape[-1] // embed.shape[-1]
            if self.slide_crop > 0 and self.slide_crop<inputs.shape[-1]:
                assert self.slide_crop % weak_patch_size == 0, f'slide_crop ({self.slide_crop}) should be divisible by weak_patch_size ({weak_patch_size})'
                assert self.slide_stride % weak_patch_size == 0, f'slide_stride ({self.slide_stride}) should be divisible by weak_patch_size ({weak_patch_size})'
                seg_logits = self.forward_slide(inputs, batch_img_metas, embed, weak_patch_size,
                                                stride=self.slide_stride, crop_size=self.slide_crop)
            else:
                seg_logits = self.forward_feature(inputs, embed, 
                                                  logit_size=batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, return_prob=return_prob), weak_pred, weak_logits #, mid_feats

    def postprocess_result(self, seg_logits, return_prob=False): #, data_samples):
        # print(f'seg_logits:{seg_logits.shape}')
        # batch_size = seg_logits.shape[0]
        # for i in range(batch_size):

        num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
        if num_cls != num_queries:
            seg_logits = seg_logits.unsqueeze(1)
            cls_index = nn.functional.one_hot(self.query_idx)
            cls_index = cls_index.T.view(1, num_cls, num_queries, 1, 1)
            seg_logits = (seg_logits * cls_index).max(2)[0]

        seg_logits = seg_logits * self.logit_scale
        seg_logits = seg_logits.softmax(1)  # n_queries * w * h
        seg_pred = seg_logits.argmax(1) 
        seg_pred[seg_logits.max(1)[0]< self.prob_thd] = 0

        if return_prob:
            return seg_logits
        else:
            return seg_pred
    
    def get_weak_model_prediction(self, batch, return_embeddings=False):
        with autocast(): 
            outs_dict = self.vfm.predict_with_multiple_prompts(batch, return_embeddings=return_embeddings)
        
        # 1 -  mask
        logits = outs_dict['masks_list'][-1]

        # # get the largest logits among multiple prompts
        num_cls, num_queries = max(self.class_id_list) + 1, len(self.class_id_list)
        # if num_cls != num_queries:
        #     logits = logits.unsqueeze(1)
        #     cls_index = nn.functional.one_hot(self.class_id_list)
        #     cls_index = cls_index.T.view(1, num_cls, num_queries, 1, 1)
        #     logits = (logits * cls_index).max(2)[0]
        # logits = logits * self.logit_scale
        # logits = logits.softmax(1)  # n_queries * w * h

        logits = (logits*self.logit_scale).softmax(1)
        if num_cls != num_queries:
            logits = logits.unsqueeze(2)
            cls_index = nn.functional.one_hot(self.class_id_list, num_cls)
            cls_index = cls_index.view(1, num_queries, num_cls, 1, 1)
            logits = (logits * cls_index).max(1)[0]
            
        seg_pred = logits.argmax(1) 
        seg_pred[logits.max(1)[0]< self.prob_thd] = 0

        # 2 - embeddings
        if return_embeddings:
            if self.weak_feat_type == 'encoder_embed':
                embed = outs_dict['img_feats']['image_embed']
            elif self.weak_feat_type == 'preout_embed':
                embed = outs_dict['output_embeddings'][-1]
            else:
                raise ValueError(f'Wrong weak_feat_type={self.weak_feat_type}')
            return seg_pred, logits, embed
        else:
            return seg_pred, logits


    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices