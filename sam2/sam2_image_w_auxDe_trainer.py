from sam2.sam2_image_trainer import SAM2ImageTrainer as BaseTrainer


class SAM2ImageTrainerAux(BaseTrainer):
    def __init__(self, 
                 args,
                 ):
        super().__init__(args)

        assert self.model.use_auxiliary_decoder, "Model must have an auxiliary decoder."
    

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
        iou_predictions, precise_predictions = \
            self.forward_w_mask((masks>0).to(y.dtype), img_feats, text_embeddings, use_aux=(not aux))
        loss_quality = self.L1Loss(iou_predictions[y_exist==1], ious[y_exist==1]).mean() + \
                        self.L1Loss(precise_predictions[y_exist==1], precises[y_exist==1]).mean()
        loss_quality *= n_classes
        self.log(f'tr{counter_pref}_loss_qual', loss_quality, on_epoch=True, on_step=False, sync_dist=True)
        loss += loss_quality*n_classes

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
        self._do_validation(batch, batch_idx, prefix='test', use_aux=False)
        self._do_validation(batch, batch_idx, prefix='test', use_aux=True)