import h5py
import torch
import numpy as np
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer
from sam2.sam2_image_trainer import KEY_DATASAT, MULTI_MASK_KEYS

EXACT_DATASETS = ['openearthmap', 'dynamicearthnet', 'dw']


class SAM2ImageTrainerSepAuxDe4Weak(BaseTrainer):
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
        if len(aux_decoders)>0:
            aux_list = [aux_decoders] + [[False]*len(cls_name_keys)]
        else:
            aux_list = [[False]*len(cls_name_keys)]
        for aux in aux_list:
            for auxi, (y, y_exist, n_class, prompt_token, name_key) in enumerate(zip(ys, y_exists, n_classes, prompt_tokens, cls_name_keys)): 
                text_embeddings = self._get_text_embeddings(prompt_token)

                # prediction
                masks, iou_predictions, precise_predictions, exist_predictions = \
                    self.forward_decoder(img_feats, text_embeddings, n_class, x.shape[0], use_aux=aux[auxi])
            
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
                                            from_aux_decoder=aux[auxi], log=log_insides_func,
                                            )

                loss = loss + loss_

            # training image examples  
            if aux[auxi] is False and batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
                self._log_imgs(f'tr_{name_key}', x[0], masks[0], y[0])
            
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