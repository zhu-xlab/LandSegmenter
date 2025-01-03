import h5py
import torch
import numpy as np
import torch.nn.functional as F
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer
from sam2.sam2_image_trainer import MASK_CODE


class SAM2ImageTrainerAux(BaseTrainer):
    def __init__(self, 
                 args,
                 ):
        super().__init__(args)

        assert self.model.use_auxiliary_decoder, "Model must have an auxiliary decoder."
        self.use_noise_mask_loss = args.use_noise_mask_loss            

    # ------ Training ------ #
    def training_step(self, batch, batch_idx):
        log_insides_func = False
        # optimize main branch at even batch ids; optimize aux branch at odd batch ids
        aux = False if self.global_step%2==0 else True
        aux_pref = '_aux' if aux else ''
        counter_pref = '' if aux else '_aux'

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

            # prediction
            masks, iou_predictions, precise_predictions, exist_predictions = \
                self.forward_decoder(img_feats, text_embeddings, n_class, x.shape[0], use_aux=aux)
            masks_ori = masks.detach().clone()
            if masks.shape[-2:] != y.shape[-2:]:
                masks = F.interpolate(masks, size=y.shape[-2:], mode='bilinear', align_corners=False)

            # prediction accuracies
            if if_weak_lbl:
                precises, ious =  None, None
            else:
                precises, ious = self._get_accs(masks.detach(), y)
        
            # calculate losses 
            class_weights = self.class_weight_tensors[name_key] if self.use_class_weights else None
            loss_ = self._calculate_loss(n_class, y, y_exist, ious, precises, 
                                         masks, exist_predictions, iou_predictions, precise_predictions,
                                         class_weights=class_weights,
                                         from_aux_decoder=aux, log=log_insides_func,
                                         )

            if not if_weak_lbl:
                # # # minor forward process to estimate noise mask quality # # # 
                if self.use_noise_mask_loss:
                    masks_ano, iou_predictions_ano, precise_predictions_ano, exist_predictions_ano = \
                        self.forward_decoder_w_mask((masks_ori>0).to(y.dtype), img_feats, text_embeddings, use_aux=(not aux))
                    precises_ano, ious_ano = self._get_accs(masks_ano.detach(), y)
                    loss_quality = self._calculate_loss(n_class, y, y_exist, ious_ano, precises_ano,
                                                        masks_ano, exist_predictions_ano, iou_predictions_ano, precise_predictions_ano,
                                                        class_weights=class_weights,
                                                        from_aux_decoder=(not aux), log=log_insides_func,
                                                        )
                else:
                    _, iou_predictions_ano, precise_predictions_ano, _ = \
                        self.forward_decoder_w_mask((masks_ori>0).to(y.dtype), img_feats, text_embeddings, use_aux=(not aux))
                    loss_quality = self.L1Loss(iou_predictions_ano[y_exist==1], ious[y_exist==1]).mean() + \
                                    self.L1Loss(precise_predictions_ano[y_exist==1], precises[y_exist==1]).mean()
                    loss_quality = loss_quality*n_class
                if log_insides_func:
                    self.log(f'tr{counter_pref}_loss_qual', loss_quality, on_epoch=True, on_step=False, sync_dist=True)
                loss_ = loss_ + loss_quality

            # training image examples  
            if batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
                self._log_imgs(f'tr{aux_pref}_{name_key}', x[0], masks[0], y[0])
            
            # losses.append(loss)
            loss = loss + loss_
            # self.log(f'tr{counter_pref}_loss_{name_key}', loss, on_epoch=True, on_step=True, sync_dist=True, batch_size=x.shape[0])
        
        # total loss
        # loss = sum(losses)
        self.log(f'tr{counter_pref}_loss', loss, on_epoch=True, on_step=False, sync_dist=True) #, batch_size=x.shape[0])
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        self._do_validation(batch, batch_idx, prefix='val', use_aux=False)
        self._do_validation(batch, batch_idx, prefix='val', use_aux=True)
