import h5py
import torch
import wandb
import datetime
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer
from sam2.sam2_image_trainer import MASK_CODE, MULTI_MASK_KEYS

# from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from kornia.losses import (                   # type: ignore
    dice_loss, binary_focal_loss_with_logits, focal_loss,
    FocalLoss, DiceLoss
)


class SAM2ImageTrainerAuxUncScaleW(BaseTrainer):
    def __init__(self, 
                 args,
                 ):
        super().__init__(args)

        assert self.model.use_auxiliary_decoder, "Model must have an auxiliary decoder."
        # self.use_noise_mask_loss = args.use_noise_mask_loss
        # rampup parameters for dynamically setting weak weight
        self.rampup_end = args.weak_weight_rampup_end
        self.rampup_start = args.weak_weight_rampup_start
        self.rampup_length = self.rampup_end - self.rampup_start
        self.weak_weight_final = args.weak_weight_final
        self.unc_lambda = args.weak_unc_lambda
        self.no_weak_input = args.no_weak_input
        self.weak_detach = args.weak_detach
        # for rescale class weights
        self.weight_pos_mean = args.weight_pos_mean
        self.weight_neg_mean = args.weight_neg_mean
        self.cls_w_means = {}
        self.cls_w_sum_update, self.cls_n_samples_update = {}, {}
        self.cls_keys = []
        for d in args.dataset_name:
            if d in MULTI_MASK_KEYS:
                self.cls_keys += MULTI_MASK_KEYS[d]
            else:
                self.cls_keys.append(d)
        # test settings
        self.test_only = args.test_only
        if self.test_only:
            self.test_save_path = args.test_save_path
            self.y_exist = []
            self.ts_masks, self.ts_masks_aux = [], []
            self.ts_iou, self.ts_iou_aux, self.ts_iou_pred, self.ts_iou_pred_aux = [], [], [], []
            self.ts_acc, self.ts_acc_aux, self.ts_acc_pred, self.ts_acc_pred_aux = [], [], [], []
            self.ts_exist, self.ts_exist_aux = [], []
            self.ts_iou_pred_ano, self.ts_iou_pred_aux_ano = [], []
            self.ts_acc_pred_ano, self.ts_acc_pred_aux_ano = [], []

    
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        t1 = datetime.datetime.now().replace(microsecond=0)
        self.weak_weight = self.weak_weight_final*self._get_weak_loss_weight(self.current_epoch)
        if len(self.cls_n_samples_update)>0:
            if len(self.cls_keys) != len(self.cls_n_samples_update):
                self.cls_keys = [k for k in self.cls_keys if k in self.cls_n_samples_update]
            # # if self.trainer.global_rank==0:
            # #     print(f"EP{self.current_epoch}|n_sample={self.cls_n_samples_update}")
            # for k in self.cls_keys:
            #     cls_w_means_all = self._gather_data_across_gpus(self.cls_w_means_update[k])
            #     cls_n_samples_all = self._gather_data_across_gpus(self.cls_n_samples_update[k])
            #     cls_w_means_all = torch.stack(cls_w_means_all, dim=0)
            #     cls_n_samples_all = torch.stack(cls_n_samples_all, dim=0)
            #     cls_w_means = (cls_w_means_all*cls_n_samples_all).sum(dim=0)/cls_n_samples_all.sum(dim=0)
            #     self.cls_w_means[k] = cls_w_means.detach()
            # if self.trainer.global_rank==0:
            #     print(f"cls_n_samples={self.cls_n_samples_update}")
            # self.cls_w_means_update.clear()
            # self.cls_n_samples_update.clear()
            # t2 = datetime.datetime.now().replace(microsecond=0)
            # if self.current_epoch<5 and self.trainer.global_rank==0:
            #     print(f"Train_start-{t2-t1}-{self.trainer.global_rank}: cls_w_means={self.cls_w_means}")
            
            # with torch.no_grad():   
            #     for k in self.cls_keys:
            #         cls_w_sum_all = self._gather_data_across_gpus(self.cls_w_sum_update[k])
            #         cls_n_samples_all = self._gather_data_across_gpus(self.cls_n_samples_update[k])
            #         if self.trainer.global_rank==0:
            #             print(f"EP{self.current_epoch}|cls_n_samples_all={cls_n_samples_all}")
            #             print(f"EP{self.current_epoch}|cls_w_means_all={cls_w_means_all}")
            #         cls_w_sum_all = torch.cat(cls_w_sum_all, dim=0)
            #         cls_n_samples_all = torch.cat(cls_n_samples_all, dim=0)
            #         cls_w_means = cls_w_sum_all.sum(dim=0) / cls_n_samples_all.sum(dim=0)
            #         self.cls_w_means[k] = cls_w_means.detach()
            #         if self.trainer.global_rank==0:
            #             print(f"k={k}|cls_n_samples={self.cls_n_samples_update[k]}")  
            # self.cls_w_means_update.clear()
            # self.cls_n_samples_update.clear()
            for k in self.cls_keys:
                cls_w_sum = self.cls_w_sum_update[k]
                cls_n_samples = self.cls_n_samples_update[k]
                torch.distributed.all_reduce(cls_w_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cls_n_samples, op=torch.distributed.ReduceOp.SUM)
                self.cls_sums[k] = cls_w_sum / cls_n_samples
                print(k, self.cls_sums[k], cls_n_samples)
            self.cls_w_sum_update.clear()
            self.cls_n_samples_update.clear()


    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        log_insides_func = False

        # get data
        x, ys, waves, y_exists, n_classes, prompt_tokens, if_weak_lbl, cls_name_keys = \
            self._get_data(batch, return_single=False)

        # # # major forward process # # #            
        # image embeddings
        img_feats = self._get_image_embed(x, waves, True)
        # for uncertainty branch
        if self.weak_detach:
            img_feats_unc = {}
            for k, v in img_feats.items():
                if isinstance(v, torch.Tensor):
                    img_feats_unc[k] = v.detach()
                else:
                    img_feats_unc[k] = [v_i.detach() for v_i in v]
        else:
            img_feats_unc = img_feats

        # make predictions and calculate losses
        loss = torch.tensor(0., device=self.device)
        for y, y_exist, n_class, prompt_token, name_key in zip(ys, y_exists, n_classes, prompt_tokens, cls_name_keys): 
            text_embeddings = self._get_text_embeddings(prompt_token)

            # classification predictions
            masks, iou_predictions, precise_predictions, exist_predictions = \
                self.forward_decoder(img_feats, text_embeddings, n_class, x.shape[0])
            
            # # dropout
            # if self.dropout_rate>0:
            #     image_embed = img_feats['image_embed']
            #     image_embed_dropout = []
            #     for _ in range(self.dropout_n):
            #         image_embed_dropout_ = self.dropout(image_embed)
            #         image_embed_dropout.append(image_embed_dropout_)
            #     img_feats['image_embed'] = sum(image_embed_dropout) / self.dropout_n

            if if_weak_lbl:
                precises, ious =  None, None
                # uncertainty predictions
                hetero_uncs = self._get_hetero_uncertainty(y, img_feats_unc, text_embeddings)
            else:
                precises, ious = self._get_accs(masks.detach(), y)
        
            # calculate losses 
            # part 1: normal loss for main branch only
            class_weights = self.class_weight_tensors[name_key] if self.use_class_weights else None
            loss_ = self._calculate_loss(n_class, y, y_exist, ious, precises, 
                                         masks, exist_predictions, iou_predictions, precise_predictions,
                                         class_weights=class_weights, 
                                         from_aux_decoder=False, log=log_insides_func,
                                         )
            
            # part 2: weak label loss with estimated uncertainty from aux decoder
            y_valid = (y!=MASK_CODE).unsqueeze(1).expand_as(masks)
            if if_weak_lbl: # if_weak_lbl and self.weak_weight>0:
                weights_cls = torch.exp(-hetero_uncs)
                weights_cls = torch.nan_to_num(weights_cls, nan=0.0, posinf=0.0, neginf=0.0)
                y_onehot = self._update_weight_cls_stats(weights_cls, y, name_key)
                
                # y_onehot = F.one_hot(y, num_classes=n_class)
                # y_onehot[y==MASK_CODE,:] = 0
                # y_onehot = y_onehot.permute(0, 3, 1, 2)

                if len(self.cls_w_means)>0:
                    weights_cls = self._rescale_weight_cls(weights_cls, y_onehot, name_key)
                
                # weights_avg = weights_cls.mean(dim=1)
                # weights_avg = torch.zeros_like(y, dtype=torch.float32, device=x.device)
                # for c in range(n_class):
                #     weights_avg[y==c] = weights_cls[:,c][y==c]
                weights_avg = (weights_cls*y_onehot).sum(dim=1)
                
                weak_loss_ = self._calculate_loss(n_class, y, y_exist, ious, precises, 
                                                  masks, exist_predictions, iou_predictions, precise_predictions,
                                                  class_weights=class_weights, weights_list=[weights_cls, weights_avg],
                                                  from_aux_decoder=False, log=log_insides_func,
                                                  )
                unc_loss_ = self.unc_lambda*hetero_uncs[y_valid].mean()
                loss_ = (1-self.weak_weight)*loss_ + self.weak_weight*(weak_loss_+unc_loss_)
            elif not if_weak_lbl:
                # for exact labels
                y_pred = (masks.detach()>0).to(y.dtype)
                hetero_unc_gt = self._get_hetero_uncertainty(y_pred, img_feats_unc, text_embeddings)
                unc_loss = self.unc_lambda*self._get_uncertainty_loss_for_predictions(y_pred, y, hetero_unc_gt)
                if log_insides_func:
                    self.log('tr_loss_gt_unc', unc_loss, on_epoch=True, on_step=False, sync_dist=True)
                loss_ = loss_ + unc_loss

            # training image examples  
            if batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
                self._log_imgs(f'tr_{name_key}', x[0], masks[0], y[0], hetero_uncs[0] if if_weak_lbl else hetero_unc_gt[0])
            
            # losses.append(loss)
            loss = loss + loss_
            # self.log(f'tr{counter_pref}_loss_{name_key}', loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=x.shape[0])
        
        # total loss
        self.log(f'tr_loss', loss, on_epoch=True, on_step=False, sync_dist=True) #, batch_size=x.shape[0])
        
        return loss


    def validation_step(self, batch, batch_idx):
        self._do_validation(batch, batch_idx, prefix='val', use_aux=False)
        # self._do_validation(batch, batch_idx, prefix='val', use_aux=True)


    def test_step(self, batch, batch_idx):
        # self._do_validation(batch, batch_idx, prefix='test', use_aux=False)
        # self._do_validation(batch, batch_idx, prefix='test', use_aux=True)

        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens, _, _ = self._get_data(batch)

        # prediction
        masks, iou_predictions, precise_predictions, existance_predictions, img_feats, text_embeddings = \
            self.forward(x, prompt_tokens, waves, use_aux=False)
        masks_aux, iou_predictions_aux, precise_predictions_aux, existance_predictions_aux = \
            self.forward_decoder(img_feats, text_embeddings, n_classes, x.shape[0], use_aux=True)
        
        # quality assessment prediction
        _, iou_predictions_ano, precise_predictions_ano, _ = \
            self.forward_decoder_w_mask((masks>0).to(y.dtype), img_feats, text_embeddings, use_aux=True)
        _, iou_predictions_aux_ano, precise_predictions_aux_ano, _ = \
            self.forward_decoder_w_mask((masks_aux>0).to(y.dtype), img_feats, text_embeddings, use_aux=False)
        

        # accuracies of mask predictions
        accs, ious = self._get_accs(masks.detach(), y)
        accs_aux, ious_aux = self._get_accs(masks_aux.detach(), y)
        # exist_acc = self._get_exist_accuracy(existance_predictions.detach(), y_exist)
        # exist_acc_aux = self._get_exist_accuracy(existance_predictions_aux.detach(), y_exist)

        # save results
        self.y_exist.append(y_exist.cpu().numpy().astype(np.uint8))
        self.ts_masks.append(masks.detach().cpu().numpy()) #.argmax(axis=1).astype(np.uint8))
        self.ts_masks_aux.append(masks_aux.detach().cpu().numpy()) #.argmax(axis=1).astype(np.uint8))
        self.ts_iou.append(ious.cpu().numpy())
        self.ts_iou_aux.append(ious_aux.cpu().numpy())
        self.ts_iou_pred.append(iou_predictions.detach().cpu().numpy())
        self.ts_iou_pred_aux.append(iou_predictions_aux.detach().cpu().numpy())
        self.ts_acc.append(accs.cpu().numpy())
        self.ts_acc_aux.append(accs_aux.cpu().numpy())
        self.ts_acc_pred.append(precise_predictions.detach().cpu().numpy())
        self.ts_acc_pred_aux.append(precise_predictions_aux.detach().cpu().numpy())
        self.ts_exist.append((existance_predictions.detach().cpu().numpy()>0).astype(np.uint8))
        self.ts_exist_aux.append((existance_predictions_aux.detach().cpu().numpy()>0).astype(np.uint8))
        self.ts_iou_pred_ano.append(iou_predictions_ano.detach().cpu().numpy())
        self.ts_iou_pred_aux_ano.append(iou_predictions_aux_ano.detach().cpu().numpy())
        self.ts_acc_pred_ano.append(precise_predictions_ano.detach().cpu().numpy())
        self.ts_acc_pred_aux_ano.append(precise_predictions_aux_ano.detach().cpu().numpy())

        # y_exist = y_exist.to(torch.bool)
        # self.log(f'{prefix}{aux_tag}_mious_exist', ious[y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        # self.log(f'{prefix}{aux_tag}_mious_noexist', ious[~y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        # self.log(f'{prefix}{aux_tag}_mious_pred_exist', iou_predictions[y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        # self.log(f'{prefix}{aux_tag}_mious_pred_noexist', iou_predictions[~y_exist].mean(), on_epoch=True, on_step=False, sync_dist=True)
        # self.log(f'{prefix}{aux_tag}_exist_acc', exist_acc, on_epoch=True, on_step=False, sync_dist=True)
        # if batch_idx==0 and self.trainer.global_rank==0:
        #     for i in range(1):
        #         self._log_imgs(prefix+aux_tag, x[i], masks[i], y[i])
        
    
    def on_test_epoch_end(self):
        with h5py.File(self.test_save_path, 'w') as h:
            for k, v in zip(['y_exist', 'ts_masks', 'ts_masks_aux', 'ts_iou', 'ts_iou_aux', 'ts_iou_pred', 'ts_iou_pred_aux', 'ts_acc', 'ts_acc_aux', 'ts_acc_pred', 'ts_acc_pred_aux', 'ts_exist', 'ts_exist_aux', 'ts_iou_pred_ano', 'ts_iou_pred_aux_ano', 'ts_acc_pred_ano', 'ts_acc_pred_aux_ano'], 
                            [self.y_exist, self.ts_masks, self.ts_masks_aux, self.ts_iou, self.ts_iou_aux, self.ts_iou_pred, self.ts_iou_pred_aux, self.ts_acc, self.ts_acc_aux, self.ts_acc_pred, self.ts_acc_pred_aux, self.ts_exist, self.ts_exist_aux, self.ts_iou_pred_ano, self.ts_iou_pred_aux_ano, self.ts_acc_pred_ano, self.ts_acc_pred_aux_ano]):
                h.create_dataset(k, data=np.concatenate(v, axis=0))
                v.clear()


    def _get_weak_loss_weight(self, current):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if self.rampup_length == 0:
            return 1.0
        else:
            current = current - self.rampup_start
            current = np.clip(current, 0.0, self.rampup_length)
            phase = 1.0 - current / self.rampup_length
            return float(np.exp(-5.0 * phase * phase))
    

    def _get_uncertainty_loss_for_predictions(self, predictions, y, hetero_unc):
        n_classes = predictions.shape[1]
        y_valid = (y!=MASK_CODE).unsqueeze(1).expand_as(predictions)
        y_one_hot = F.one_hot(y.clamp(0, n_classes - 1), num_classes=n_classes).permute(0, 3, 1, 2)  # Reshape to [5, 3, 4, 4]
        y_one_hot = y_one_hot * y_valid
        predictions = predictions * y_valid
        considered_4loss = (y_one_hot>0)+(predictions>0)
        # correct part
        correct_part = ((y_one_hot*predictions) * hetero_unc).sum()  # hetero_unc -> 0
        # incorrect part
        incorrect_part = (32 - (y_one_hot*(1-predictions) + (1-y_one_hot)*predictions) * hetero_unc).sum()  # hetero_unc -> inf (32 because of clip)
        # loss
        loss = (correct_part + incorrect_part)/considered_4loss.sum()
        
        return loss


    def _log_imgs(self, prefix, x, mask, y, hetero_unc=None):
        n_classes = mask.shape[0]
        if n_classes==1:
            mask = (mask[0].detach().cpu().numpy()>0.5).astype(np.uint8)
        else:
            # maxp = mask.max(axis=0).values
            mask = mask.argmax(axis=0) # +1
            # mask[maxp<0.5] = 0
            mask = mask.detach().cpu().numpy().astype(np.uint8)
        class_labels = {i:str(i) for i in range(n_classes+1)}
        # y += 1
        # y[y>=MASK_CODE] = 0
        masks={'prediction':   {'mask_data': mask, 'class_labels': class_labels},
               'ground truth': {'mask_data': y.detach().cpu().numpy(), 'class_labels': class_labels}
                }
        # get rgb img
        # x = F.interpolate(x.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
        if x.shape[0]==3:
            rgb_img = x.detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[0]>3:
            rgb_img = x[:3].detach().cpu().numpy().transpose((1,2,0))
        # seg results
        mask_img = wandb.Image(rgb_img, masks=masks, caption=f'{prefix}_masks')
        log_imgs = [mask_img]
        # hetero_unc
        if hetero_unc is not None:
            unc_img = hetero_unc.detach().cpu().numpy().mean(axis=0)
            # weight_img = np.exp(-unc_img)
            unc_img = unc_img/32
            unc_img = wandb.Image(unc_img, caption=f'{prefix}_uncertainty')
            # weight_img = wandb.Image(weight_img, caption=f'{prefix}_weight')
            log_imgs.append(unc_img)
            # log_imgs.append(weight_img)
        wandb.log({prefix:log_imgs})
    
    def _get_hetero_uncertainty(self, y, img_feats, text_embeddings):
        if not self.no_weak_input:
            # print("use weak input")
            hetero_uncs_, _, _, _ = \
                        self.forward_decoder_w_mask(y, img_feats, text_embeddings, use_aux=True)
        else:
            # print("no weak input")
            # # # # debug: v3_wolblinput
            hetero_uncs_, _, _, _ = \
                        self.forward_decoder(img_feats, text_embeddings, text_embeddings.shape[0], y.shape[0], use_aux=True)
        # # # # debug: v4_wolblinput1empty
        # hetero_uncs_, _, _, _ = \
        #             self.forward_decoder(img_feats, text_embeddings, text_embeddings.shape[0], y.shape[0], use_aux=True, one_no_noise_mask_embed=True)
        if self.use_logvar_smooth:
            # print("use logvar smooth")
            hetero_uncs_ = rearrange(hetero_uncs_, 'b c h w -> (b c) 1 h w')
            hetero_uncs_ = self.model.logvar_smoother(hetero_uncs_)
            hetero_uncs_ = rearrange(hetero_uncs_, '(b c) 1 h w -> b c h w', b=y.shape[0])
        hetero_uncs = F.relu(hetero_uncs_)
        return hetero_uncs
    
    @torch.no_grad()
    def _update_weight_cls_stats(self, weights_cls, y, name_key):
        weights_cls = weights_cls.detach()
        n_class = weights_cls.shape[1]
        y_onehot = F.one_hot(y, num_classes=n_class).permute(0, 3, 1, 2) * (y!=MASK_CODE)[:,None]
        y_onehot_neg = (1-y_onehot) * (y!=MASK_CODE)[:,None]
        if name_key not in self.cls_w_sum_update:
            self.cls_w_sum_update[name_key] = torch.zeros((2, n_class), requires_grad=False, device=self.device)     # 0: pos, 1: neg
            self.cls_n_samples_update[name_key] = torch.zeros((2, n_class), requires_grad=False, device=self.device)
            # self.cls_w_means_update[name_key] = [0.]*n_class
            # self.cls_n_samples_update[name_key] = [0.]*n_class
        self.cls_n_samples_update[name_key][0] = self.cls_n_samples_update[name_key][0] + y_onehot.sum(dim=(0,2,3))
        self.cls_n_samples_update[name_key][1] = self.cls_n_samples_update[name_key][1] + y_onehot_neg.sum(dim=(0,2,3))
        self.cls_w_sum_update[name_key][0] = self.cls_w_sum_update[name_key][0] + (weights_cls*y_onehot).sum(dim=(0,2,3))
        self.cls_w_sum_update[name_key][1] = self.cls_w_sum_update[name_key][1] + (weights_cls*y_onehot_neg).sum(dim=(0,2,3))

        return y_onehot 

        # for c in range(n_class):
        #     w = weights_cls[:,c]
        #     n_pos = (y==c).sum()
        #     n_neg = ((y!=c)*(y!=MASK_CODE)).sum()
        #     w_sum_pos = w[y==c].sum()
        #     w_sum_neg = w[(y!=c)*(y!=MASK_CODE)].sum()
        #     if n_pos>0:
        #         self.cls_w_sum_update[name_key][0,c] = self.cls_w_sum_update[name_key][0,c] + w_sum_pos
        #             # (self.cls_w_means_update[name_key][0,c]*self.cls_n_samples_update[name_key][0,c] + w_mean_pos*n_pos)/(self.cls_n_samples_update[name_key][0,c]+n_pos)
        #         self.cls_n_samples_update[name_key][0,c] = self.cls_n_samples_update[name_key][0,c] + n_pos
        #     if n_neg>0:
        #         self.cls_w_sum_update[name_key][1,c] = self.cls_w_sum_update[name_key][1,c] + w_sum_neg
        #         # self.cls_w_means_update[name_key][1,c] = \
        #         #     (self.cls_w_means_update[name_key][1,c]*self.cls_n_samples_update[name_key][1,c] + w_mean_neg*n_neg)/(self.cls_n_samples_update[name_key][1,c]+n_neg)
        #         self.cls_n_samples_update[name_key][1,c] = self.cls_n_samples_update[name_key][1,c] + n_neg
    

    def _rescale_weight_cls(self, weights_cls, y_onehot, name_key):
        rescale_pos = y_onehot * (self.weight_pos_mean-self.cls_w_means[name_key][0,:][None,:,None,None])
        rescale_neg = (1-y_onehot) * (self.weight_neg_mean-self.cls_w_means[name_key][1,:][None,:,None,None])
        rescale_matrix = rescale_pos + rescale_neg
        weights_cls = torch.clip(weights_cls + rescale_matrix, 0, 1)
        return weights_cls
        # rescale_matrix = torch.zeros_like(weights_cls, device=y_onehot.device, requires_grad=False)
        # for c in range(n_class):
        #     rescale_matrix[:,c] = torch.where(y==c, 
        #                                       self.weight_pos_mean-self.cls_w_means[name_key][0,c],
        #                                       self.weight_neg_mean-self.cls_w_means[name_key][1,c])
        # weights_cls = torch.clip(weights_cls + rescale_matrix, 0, 1)
        # return weights_cls


    # def _gather_data_across_gpus(self, local_data) -> list[torch.Tensor]:
    #     gathered_data = [None] * self.trainer.world_size  # Placeholder for data from each process
    #     torch.distributed.all_gather_object(gathered_data, local_data)  # Gather data from all processes
    #     # move all the tensors to the current device
    #     # current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    #     # gathered_data = [data.to(current_device) for data in gathered_data]
    #     gathered_data = [data.to(local_data.device) for data in gathered_data]
    #     return gathered_data # type: ignore
    
    def _gather_data_across_gpus(self, local_data: torch.Tensor) -> torch.Tensor:
        gathered_data = [torch.zeros_like(local_data) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gathered_data, local_data)
        gathered_data = torch.stack(gathered_data, dim=0)
        return gathered_data
