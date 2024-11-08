import h5py
import torch
import wandb
import numpy as np
import torch.nn.functional as F
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer
from sam2.sam2_image_trainer import MASK_CODE


class SAM2ImageTrainerAuxUnc(BaseTrainer):
    def __init__(self, 
                 args,
                 ):
        super().__init__(args)

        assert self.model.use_auxiliary_decoder, "Model must have an auxiliary decoder."
        self.use_noise_mask_loss = args.use_noise_mask_loss
        # rampup parameters for dynamically setting weak weight
        self.rampup_end = args.weak_weight_rampup_end
        self.rampup_start = args.weak_weight_rampup_start
        self.rampup_length = self.rampup_end - self.rampup_start
        self.weak_weight_final = args.weak_weight_final
        self.unc_lambda = args.weak_unc_lambda
        # self.dropout_rate = args.dropout_rate
        # if self.dropout_rate>0: self.dropout = torch.nn.Dropout(p=self.dropout_rate)
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
        self.weak_weight = self.weak_weight_final*self._get_weak_loss_weight(self.current_epoch)
        
    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        log_insides_func = False

        # get data
        x, ys, waves, y_exists, n_classes, prompt_tokens, if_weak_lbl, cls_name_keys = \
            self._get_data(batch, return_single=False)

        # # # major forward process # # #            
        # image embeddings
        img_feats = self._get_image_embed(x, waves, True)

        # make predictions and calculate losses
        loss = torch.tensor(0., device=self.device)
        for y, y_exist, n_class, prompt_token, name_key in zip(ys, y_exists, n_classes, prompt_tokens, cls_name_keys): 
            text_embeddings = self._get_text_embeddings(prompt_token)

            # classification predictions
            masks, iou_predictions, precise_predictions, exist_predictions = \
                self.forward_decoder(img_feats, text_embeddings, n_class, x.shape[0])
            
            # # TBD: dropout
            # if self.dropout_rate>0:
            #     img_feats_list = []
            #     for i in range(self.dropout_n):
            #         img_feats_ = F.dropout(img_feats)
            #         img_feats_list.append(img_feats_)

            if if_weak_lbl:
                precises, ious =  None, None
                # uncertainty predictions
                hetero_uncs_, _, _, _ = \
                    self.forward_decoder_w_mask(y, img_feats, text_embeddings, use_aux=True)
                hetero_uncs = F.relu(hetero_uncs_)
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
            if if_weak_lbl and self.weak_weight>0:
                weak_loss_ = self._calculate_loss(n_class, y, y_exist, ious, precises, 
                                                  masks, exist_predictions, iou_predictions, precise_predictions,
                                                  class_weights=class_weights, hete_uncertainty=hetero_uncs,
                                                  from_aux_decoder=False, log=log_insides_func,
                                                  )
                unc_loss_ = self.unc_lambda*hetero_uncs[y_valid].mean()
                loss_ = (1-self.weak_weight)*loss_ + self.weak_weight*(weak_loss_+unc_loss_)
            elif not if_weak_lbl:
                # for exact labels
                y_pred = (masks.detach()>0).to(y.dtype)
                hetero_unc_gt, _, _, _ = \
                    self.forward_decoder_w_mask(y_pred, img_feats, text_embeddings, use_aux=True)
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