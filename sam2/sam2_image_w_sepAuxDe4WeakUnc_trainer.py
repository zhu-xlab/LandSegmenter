import h5py
import torch
import numpy as np
import torch.nn.functional as F
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer
from sam2.sam2_image_trainer import KEY_DATASAT, MULTI_MASK_KEYS, MASK_CODE

EXACT_DATASETS = ['openearthmap', 'dynamicearthnet', 'dw']


class SAM2ImageTrainerSepAuxDe4WeakUnc(BaseTrainer):
    def __init__(self, 
                 args,
                 ):
        # create dataset name and auxiliary decoder index mapping
        ind = 0
        aux_decoder_indexes = {}
        for k in args.dataset_name:
            if k not in EXACT_DATASETS:
                if k in MULTI_MASK_KEYS:
                    for subk in MULTI_MASK_KEYS[k]:
                        aux_decoder_indexes[subk] = ind
                        ind += 1
                else:
                    aux_decoder_indexes[k] = ind
                    ind += 1
        # number of auxiliary decoders
        args.use_aux_decoder = len(aux_decoder_indexes)
        super().__init__(args)
        assert self.model.use_auxiliary_decoder, "Model must have an auxiliary decoder."
        # self.use_noise_mask_loss = args.use_noise_mask_loss
        self.aux_decoder_indexes = aux_decoder_indexes
        # rampup parameters for dynamically setting weak weight
        self.rampup_end = args.weak_weight_rampup_end
        self.rampup_start = args.weak_weight_rampup_start
        self.rampup_length = self.rampup_end - self.rampup_start
        self.weak_weight_final = args.weak_weight_final
        self.unc_lambda = args.weak_unc_lambda
        # test only settings
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
        dataset = batch[KEY_DATASAT][0]
        if dataset in EXACT_DATASETS:
            aux_decoders = []
        else:
            aux_decoders = [self.aux_decoder_indexes[k] for k in cls_name_keys]

        # # # major forward process # # #            
        # image embeddings and high-resolution features from the main decoder
        img_feats = self._get_image_embed(x, waves, aux_decoders)

        # make predictions and calculate losses
        loss = torch.tensor(0., device=self.device)
        for aux, y, y_exist, n_class, prompt_token, name_key in zip(aux_decoders, ys, y_exists, n_classes, prompt_tokens, cls_name_keys): 
            text_embeddings = self._get_text_embeddings(prompt_token)

            # classification prediction using main decoder
            masks, iou_predictions, precise_predictions, exist_predictions = \
                self.forward_decoder(img_feats, text_embeddings, n_class, x.shape[0], use_aux=False)

            # uncertainty predictions using auxiliary decoder
            if if_weak_lbl:
                precises, ious =  None, None
                # uncertainty predictions
                hetero_uncs_, _, _, _ = \
                    self.forward_decoder_w_mask(y, img_feats, text_embeddings, use_aux=aux)
                hetero_uncs = F.relu(hetero_uncs_, inplace=False)
                # hetero_uncs[hetero_uncs.isnan()+hetero_uncs.isinf()] = 32.0
            else:
                # if it's exact labels, skip
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
                weights = torch.exp(-hetero_uncs)
                weak_loss_ = self._calculate_loss(n_class, y, y_exist, ious, precises, 
                                                  masks, exist_predictions, iou_predictions, precise_predictions,
                                                  class_weights=class_weights, weights=weights,
                                                  from_aux_decoder=False, log=log_insides_func,
                                                  )
                unc_loss_ = self.unc_lambda*hetero_uncs[y_valid].mean()
                loss_ = (1-self.weak_weight)*loss_ + self.weak_weight*(weak_loss_+unc_loss_)

            # training image examples  
            if batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
                self._log_imgs(f'tr_{name_key}', x[0], masks[0], y[0])
            
            # accumulate loss
            loss = loss + loss_
        
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