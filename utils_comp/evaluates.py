# -*- coding: utf-8 -*-
"""
Evaluation utils

Created on Thu Jun 22 17:28:26 2023

@author: liu_ch
"""
import torch
import numpy as np
import segmentation_models_pytorch as smp

REDUCE = 'batchwise' # 'imagewise' # 'batchwise'


# # # # # # # # # # Metrics calculation # # # # # # # # # #
def cal_OA(output, target):
    # calculate OAs
    judge_matr = ((output-target)==0)
    oa_score = (judge_matr).sum()/output.numel()
    return oa_score


def cal_acc_metrics_multiclass(output, target, num_classes, reduce=REDUCE): # reduce: imagewise/batchwise
    mode = 'binary' if num_classes==1 else 'multiclass'  
    OA = cal_OA(output, target)
    # first compute statistics for true positives, false positives, false negative and true negative "pixels"
    if reduce=='batchwise':
        # flatten
        output = output.view(1, -1)
        target = target.view(1, -1)
        if num_classes==1: output, target = output[:,None], target[:,None]
        # compute statistics for true positives, false positives, false negative and true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, num_classes=num_classes, mode=mode)
        # Sum true positive, false positive, false negative and true negative pixels over all images for each label, then compute score for each label separately and average labels scores. This does not take label imbalance into account.
        ious = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").squeeze()
        miou = ious.sum()/num_classes
        f1s = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").squeeze()
        mf1 = f1s.sum()/num_classes
        prs = smp.metrics.precision(tp, fp, fn, tn, reduction="none").squeeze()
        mpr = prs.sum()/num_classes
        rcs = smp.metrics.recall(tp, fp, fn, tn, reduction="none").squeeze()
        mrc = rcs.sum()/num_classes
    elif reduce=='imagewise':
        if num_classes==1: output, target = output[:,None], target[:,None]
        tp, fp, fn, tn = smp.metrics.get_stats(output, target, num_classes=num_classes, mode=mode)
        ious_all = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
        ious = ious_all.mean(0)
        miou = ious.sum()/num_classes
        f1s_all = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none")
        f1s = f1s_all.mean(0)
        mf1 = f1s.sum()/num_classes
        prs_all = smp.metrics.precision(tp, fp, fn, tn, reduction="none")
        prs = prs_all.mean(0)
        mpr = prs.sum()/num_classes
        rcs_all = smp.metrics.recall(tp, fp, fn, tn, reduction="none")
        rcs = rcs_all.mean(0)
        mrc = rcs.sum()/num_classes
    
    if num_classes==1:
        metrics = {'iou':miou,
                   'f1':mf1,
                   'precise':mpr,
                   'recall':mrc,
                   'oa':OA}
    else:
        metrics = {'iou':torch.hstack((ious,miou)),
                   'f1':torch.hstack((f1s,mf1)),
                   'precise':torch.hstack((prs,mpr)),
                   'recall':torch.hstack((rcs,mrc)),
                   'oa':OA}
    return metrics
            

# # # # # # # # # # For metric recording # # # # # # # # # #
def add_class_metrics_to_val_dict(val_metrics, caccs, mname='iou', mean=True):
    if type(caccs)==float:
        # for 1 class case
        val_metrics[mname] = caccs
    else:
        # for multi-class case
        n_classes = caccs.shape[0]-1 if mean else caccs.shape[0]
        if type(caccs) == np.ndarray:
            for i in range(n_classes):
                val_metrics[f'{mname}_{i}'] = caccs[i]
            if mean: val_metrics[f'm{mname}'] = caccs[-1]
        else:
            for i in range(n_classes):
                val_metrics[f'{mname}_{i}'] = caccs[i].item()
            if mean: val_metrics[f'm{mname}'] = caccs[-1].item()
    
    return val_metrics

def add_suffix_to_metric_dict(metrics_dict, suffix, new_dict=None):
    if new_dict is None: new_dict = {}
    for k in metrics_dict:
        new_dict[f'{suffix}_{k}'] = metrics_dict[k]
    
    return new_dict


def epoch_log_dict(tr_dict, accs, batch_in_epoch, n_classes, suffix):
    acc_metrics = {'oa': accs['oa'].item()/batch_in_epoch}
    if n_classes>1:
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['iou']/batch_in_epoch, mname='iou')
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['f1']/batch_in_epoch, mname='f1')
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['pr']/batch_in_epoch, mname='precise')
        acc_metrics = add_class_metrics_to_val_dict(acc_metrics, accs['rc']/batch_in_epoch, mname='recall')
    else:
        acc_metrics['iou'] = accs['iou'].item()/batch_in_epoch
        acc_metrics['f1'] = accs['f1'].item()/batch_in_epoch
        acc_metrics['precise'] = accs['pr'].item()/batch_in_epoch
        acc_metrics['recall'] = accs['rc'].item()/batch_in_epoch
    tr_dict = add_suffix_to_metric_dict(acc_metrics, suffix=suffix, new_dict=tr_dict)
    return tr_dict


# # # # # # # # # # evaluation fucntions # # # # # # # # # #  
def evaluate(net, dataloader, device, experiment='ns_labels', n_classes=None,
             gt_label='gt'):
    # set model status to validation
    net.eval()
    # eps = 1e-8
    if n_classes is None: n_classes = net.n_classes
    
    # place holder for validation metrics
    oa_score = 0.
    iou_score = torch.zeros([n_classes+1]) if n_classes>1 else 0.
    f1_score = torch.zeros([n_classes+1]) if n_classes>1 else 0.
    pr_score = torch.zeros([n_classes+1]) if n_classes>1 else 0.
    rc_score = torch.zeros([n_classes+1]) if n_classes>1 else 0.
    
    # iterate over the validation set
    for i,batch in enumerate(dataloader):
        image = batch['img'].to(device=device, dtype=torch.float32)
        if experiment=='ns_labels':
            mask_true = torch.squeeze(batch['ns'],1).to(device=device, dtype=torch.long)
        elif experiment=='gt_labels':
            mask_true = torch.squeeze(batch[gt_label],1).to(device=device, dtype=torch.long)
        else:
            raise ValueError('Evaluation label type is invalid!')
        
        # make predictions
        with torch.no_grad():
            pred_logits = net(image).squeeze(axis=1)
        
        # get labels from softmax outputs
        if n_classes>1:
            pred = pred_logits.argmax(dim=1).long()
        else:
            pred = (torch.sigmoid(pred_logits) > 0.5).long()
        
        # calculate accuracies
        metrics = cal_acc_metrics_multiclass(pred, mask_true, num_classes=n_classes)

        oa_score += metrics['oa']
        iou_score += metrics['iou']
        f1_score += metrics['f1']
        pr_score += metrics['precise']
        rc_score += metrics['recall']
    
    # congregate val results
    num_val_batches = i+1
    val_metrics = {'oa': oa_score.item()/num_val_batches}
    if n_classes>1:
        val_metrics = add_class_metrics_to_val_dict(val_metrics, 
                                                    iou_score/num_val_batches, 
                                                    mname='iou')
        val_metrics = add_class_metrics_to_val_dict(val_metrics, 
                                                    pr_score/num_val_batches, 
                                                    mname='precise')
        val_metrics = add_class_metrics_to_val_dict(val_metrics, 
                                                    rc_score/num_val_batches, 
                                                    mname='recall')
    else:
        val_metrics['iou'] = iou_score.item()/num_val_batches
        val_metrics['precise'] = pr_score.item()/num_val_batches
        val_metrics['recall'] = rc_score.item()/num_val_batches
    
    # reset model status to training
    net.train()
    
    return val_metrics


# in a batch
def evaluate_batch_once(pred_masks, true_masks, n_classes, tr_accs, 
                        suffix, batch_to_wandb=False):
    metrics = cal_acc_metrics_multiclass(pred_masks, true_masks, n_classes)
    
    # for calculating accumulated accuracies
    if len(tr_accs)>0:
        tr_accs['oa'] += metrics['oa']
        tr_accs['pr']  += metrics['precise']
        tr_accs['rc']  += metrics['recall']
        tr_accs['iou']  += metrics['iou']
        tr_accs['f1']  += metrics['f1']
    else:
        tr_accs['oa'] = metrics['oa'].clone()
        tr_accs['pr'] = metrics['precise'].clone()
        tr_accs['rc'] = metrics['recall'].clone()
        tr_accs['iou'] = metrics['iou'].clone()
        tr_accs['f1'] = metrics['f1'].clone()
    
    # create training accuracy dict for recording to wandb
    if batch_to_wandb:
        # add accs into btr_metrics       
        btr_metrics = {'oa': metrics['oa'].item()}               
        if n_classes>1:
            btr_metrics = add_class_metrics_to_val_dict(btr_metrics, metrics['iou'], mname='iou')
            btr_metrics = add_class_metrics_to_val_dict(btr_metrics, metrics['precise'], mname='precise')
            btr_metrics = add_class_metrics_to_val_dict(btr_metrics, metrics['recall'], mname='recall')
        else:
            btr_metrics['iou'] = metrics['iou'].item()
            btr_metrics['precise'] = metrics['precise'].item()
            btr_metrics['recall'] = metrics['recall'].item()
        btr_dict = add_suffix_to_metric_dict(btr_metrics, suffix=suffix)
    else:
        btr_dict = {}
        
    return tr_accs, btr_dict


def evaluate_batch(pred_masks, true_masks, n_classes, tr_accs, suffix,
                   batch_to_wandb=False, gt_masks=None, trg_accs=None):
    tr_accs, btr_dict = \
        evaluate_batch_once(pred_masks, true_masks, n_classes, tr_accs, 
                            suffix=suffix+'b', batch_to_wandb=batch_to_wandb)
    
    # calculte accs w.r.t. gt
    if gt_masks is not None:
        assert trg_accs is not None, "Please provide trg_accs!"
        tr_gt_accs, btr_gt_dict = \
            evaluate_batch_once(pred_masks, gt_masks, n_classes, trg_accs, 
                                suffix=suffix+'gb', batch_to_wandb=batch_to_wandb)
        return tr_accs, btr_dict, tr_gt_accs, btr_gt_dict
    else:
        return tr_accs, btr_dict
    
    
# in a batch for lightning
def evaluate_batch_pl(pred_logits, true_masks, n_classes, reduce=REDUCE, thresh=0.5, mask_id=None):
    if n_classes>1:
        pred_masks = pred_logits.argmax(dim=1).long()
    else:
        pred_masks = (torch.sigmoid(pred_logits) > thresh).long()
    
    if mask_id is None:
        metrics = cal_acc_metrics_multiclass(pred_masks, true_masks, n_classes, reduce)
    else:
        pred_masks_select = pred_masks[true_masks!=mask_id]
        true_masks_select = true_masks[true_masks!=mask_id]
        metrics = cal_acc_metrics_multiclass(pred_masks_select, true_masks_select, n_classes, reduce='batchwise')
        
    return metrics


def get_confuse_matrix_batch_pl(pred_logits, true_masks, n_classes, thresh=0.5, mask_id=None):
    if n_classes>1:
        pred_masks = pred_logits.argmax(dim=1).long()
    else:
        pred_masks = (torch.sigmoid(pred_logits) > thresh).long()
    
    if mask_id is None:
        pred_masks = pred_masks.view(1, -1)
        true_masks = true_masks.view(1, -1)
    else:
        pred_masks = pred_masks[true_masks!=mask_id][None]
        true_masks = true_masks[true_masks!=mask_id][None]
    if n_classes==1: pred_masks, true_masks = pred_masks[:,None], true_masks[:,None]
    # compute statistics for true positives, false positives, false negative and true negative "pixels"
    if n_classes==1:
        mode = 'binary'
        pred_masks, true_masks = pred_masks[:,None], true_masks[:,None]
    else:
        mode = 'multiclass' 
    tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, true_masks, num_classes=n_classes, mode=mode)
    metrics = {'tp':tp, 'fp':fp, 'fn':fn, 'tn':tn}
        
    return metrics


def get_batch_acc_from_confuse_matrix(stats, num_classes):
    tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
    # Sum true positive, false positive, false negative and true negative pixels over all images for each label, then compute score for each label separately and average labels scores. This does not take label imbalance into account.
    ious = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none").squeeze()
    miou = ious.sum()/num_classes
    f1s = smp.metrics.f1_score(tp, fp, fn, tn, reduction="none").squeeze()
    mf1 = f1s.sum()/num_classes
    prs = smp.metrics.precision(tp, fp, fn, tn, reduction="none").squeeze()
    mpr = prs.sum()/num_classes
    rcs = smp.metrics.recall(tp, fp, fn, tn, reduction="none").squeeze()
    mrc = rcs.sum()/num_classes
    
    if num_classes==1:
        OA = (tp.sum()+tn.sum())/(tp.sum()+fp.sum()+fn.sum()+tn.sum())
    else:
        OA = tp.sum()/(tp.squeeze()[0]+fp.squeeze()[0]+fn.squeeze()[0]+tn.squeeze()[0])
    
    if num_classes==1:
        metrics = {'iou':miou,
                   'f1':mf1,
                   'precise':mpr,
                   'recall':mrc,
                   'oa':OA}
    else:
        metrics = {'iou':torch.hstack((ious,miou)),
                   'f1':torch.hstack((f1s,mf1)),
                   'precise':torch.hstack((prs,mpr)),
                   'recall':torch.hstack((rcs,mrc)),
                   'oa':OA}
    return metrics