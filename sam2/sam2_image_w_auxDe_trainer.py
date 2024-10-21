import h5py
import numpy as np
from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer


class SAM2ImageTrainerAux(BaseTrainer):
    def __init__(self, 
                 args,
                 ):
        super().__init__(args)

        assert self.model.use_auxiliary_decoder, "Model must have an auxiliary decoder."
        self.use_noise_mask_loss = args.use_noise_mask_loss
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
        # optimize main branch at even batch ids; optimize aux branch at odd batch ids
        aux = False if self.global_step%2==0 else True
        aux_pref = '_aux' if aux else ''
        counter_pref = '' if aux else '_aux'

        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens = self._get_data(batch)

        # # # major forward process # # #            
        # prediction
        masks, iou_predictions, precise_predictions, exist_predictions, img_feats, text_embeddings = \
            self.forward(x, prompt_tokens, waves, use_aux=aux)
        
        # prediction accuracies
        precises, ious = self._get_accs(masks.detach(), y)
        
        # calculate losses 
        loss = self._calculate_loss(n_classes, y, y_exist, ious, precises, 
                        masks, exist_predictions, iou_predictions, precise_predictions,
                        from_aux_decoder=aux,
                        )

        # # # minor forward process to estimate noise mask quality # # # 
        if self.use_noise_mask_loss:
            masks_ano, iou_predictions_ano, precise_predictions_ano, exist_predictions_ano = \
               self.forward_decoder_w_mask((masks.detach()>0).to(y.dtype), img_feats, text_embeddings, use_aux=(not aux))
            precises_ano, ious_ano = self._get_accs(masks_ano.detach(), y)
            loss_quality = self._calculate_loss(n_classes, y, y_exist, ious_ano, precises_ano,
                                                masks_ano, exist_predictions_ano, iou_predictions_ano, precise_predictions_ano,
                                                from_aux_decoder=(not aux),
                                                )
        else:
            _, iou_predictions_ano, precise_predictions_ano, _ = \
                self.forward_decoder_w_mask((masks.detach()>0).to(y.dtype), img_feats, text_embeddings, use_aux=(not aux))
            loss_quality = self.L1Loss(iou_predictions_ano[y_exist==1], ious[y_exist==1]).mean() + \
                            self.L1Loss(precise_predictions_ano[y_exist==1], precises[y_exist==1]).mean()
            loss_quality = loss_quality*n_classes
        self.log(f'tr{counter_pref}_loss_qual', loss_quality, on_epoch=True, on_step=False, sync_dist=True)
        loss += loss_quality

        # training image examples  
        if batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
            self._log_imgs(f'tr{aux_pref}', x[0], masks[0], y[0])
        
        return loss
    

    # def _first_stage_train_w_exact_labels(self, batch, batch_idx):
    #     # optimize main branch at even batch ids; optimize aux branch at odd batch ids
    #     aux = False if self.global_step%2==0 else True
    #     aux_pref = '_aux' if aux else ''

    #     # get data
    #     x, y, waves, y_exist, n_classes, prompt_tokens = self._get_data(batch)
        
    #     # prediction
    #     masks, iou_predictions, precise_predictions, exist_predictions, img_feats, text_embeddings = \
    #         self.forward(x, prompt_tokens, waves, aux=aux)
        
    #     # prediction accuracies
    #     precises, ious = self._get_accs(masks.detach(), y)
        
    #     # calculate losses 
    #     loss = self._calculate_loss(n_classes, y, y_exist, ious, precises, 
    #                     masks, exist_predictions, iou_predictions, precise_predictions,
    #                     )
        
    #     # training image examples  
    #     if batch_idx<2 and self.trainer.global_rank==0 and self.current_epoch%self.display_inv==0:
    #         self._log_imgs(f'tr{aux_pref}', x[0], masks[0], y[0])


    def validation_step(self, batch, batch_idx):
        self._do_validation(batch, batch_idx, prefix='val', use_aux=False)
        self._do_validation(batch, batch_idx, prefix='val', use_aux=True)


    def test_step(self, batch, batch_idx):
        # self._do_validation(batch, batch_idx, prefix='test', use_aux=False)
        # self._do_validation(batch, batch_idx, prefix='test', use_aux=True)

        # get data
        x, y, waves, y_exist, n_classes, prompt_tokens = self._get_data(batch)

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
        self.ts_masks.append(masks.detach().cpu().numpy().argmax(axis=1).astype(np.uint8))
        self.ts_masks_aux.append(masks_aux.detach().cpu().numpy().argmax(axis=1).astype(np.uint8))
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