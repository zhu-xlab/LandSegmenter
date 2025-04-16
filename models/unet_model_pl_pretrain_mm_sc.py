import os
import sys
import wandb
# appending self-defined package path
sys.path.append('../Codes')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# import segmentation models pytorch package
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from segmentation_models_pytorch.losses._functional import soft_dice_score

# import lightening
import pytorch_lightning as pl

# self-defined modules
import utils.evaluates as evl
from models.multimodel.unet_midFusion import Unet_midF_2branch, Unet_midF_3branch
from models.multimodel.unet_lateFusion import Unet_lateF_2branch, Unet_lateF_3branch
from utils.utils import sigmoid_rampup, linear_rampup


KEY_IMG = 'img'
KEY_GT = 'gt'
KEY_NS = 'ns'
KEY_METRICS = ['iou', 'f1', 'precise', 'recall', 'oa']


# # # # for model loading # # # # 
def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd

def send_optim_tensor_gpu(optimizer,device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == 'step': continue
                state[k] = v.to(device=device)
    return
# # # # for model loading # # # # 
    
    
class Unet_mm(pl.LightningModule):
    # 1. training setup
    def __init__(self, args):
        """
        Initializes a UNet model with the specified architecture and loss functions.

        Args:
            args: An object containing the following attributes:
                - mt: A string representing the name of the encoder to use (e.g. mobilenet_v2 or efficientnet-b7).
                - resume: A string representing the path to a checkpoint file to resume training from, or one of the following strings: 'imagenet', 'ssl', 'swsl' to use pre-trained weights for encoder initialization.
                - prt: A boolean indicating whether the resumed model is a pre-trained one for initialization or for continued training.
                - n_channels: An integer representing the number of input channels (e.g. 1 for gray-scale images, 3 for RGB, etc.).
                - n_classes: An integer representing the number of output channels (i.e. the number of classes in the dataset).
                - loss_type: A string representing the type of loss function(s) to use, where 'c' stands for Cross Entropy, 'd' stands for Dice, and 'f' stands for Focal loss.
                - if_mask: A boolean indicating whether the dataset contains masks.
                - experiment: A string representing the type of experiment label to use, where 'ns_labels' stands for non-seismic labels and 'gt_labels' stands for ground truth labels.
                - gt_avail: A boolean indicating whether ground truth labels are available when training with noisy labels.
                - cal_tr: A boolean indicating whether to calculate training metrics.
                - opt: A string representing the optimizer to use (e.g. Adam, SGD, etc.).
                - lr: A float representing the learning rate.
                - schd: A string representing the learning rate scheduler to use (e.g. StepLR, CosineAnnealingLR, etc.).
                - batch_to_wandb: An integer representing the number of batches to log to Weights & Biases.
        """
        super().__init__()
        # pl.seed_everything(args.seed)
        self.last_epoch = 0
        self.batch_wise = True #False  # whether calculate metrics batch-wisely or as a whole
        # # # # model architecture # # # #
        self.n_classes = args.n_classes
        self.n_channels_1 = 2 # for s1
        self.n_channels_2 = args.n_channels
        self.use_com_feat = args.use_com_feat
        self._get_model(args)

        # # # # segmentation loss # # # #
        self.cl_type = args.cl_type
        self.cl_weight = args.cl_weight
        self._get_loss(args.n_classes, args.loss_type, args.fuse_type, args.if_mask, args.ls)
        
        # # # # label smoothing settings # # # #
        self.ls = args.ls
        self.lsf = args.lsf
        
        # # # # sample selection settings # # # #
        self.sc = args.sc
        self.sc_rmup_len = args.sc_rmup_len # sample selection
        self.sc_conf_prop = args.sc_conf_prop
        self.sc_unc_prop = args.sc_unc_prop
        if args.sc_rmup_func=='none':
            self.sc_rmup_func = lambda x: 1.0
        else:
            self.sc_rmup_func = linear_rampup if args.sc_rmup_func=='linear' else sigmoid_rampup
        
        # # # # other hyperparameters # # # # 
        self.opt = args.opt
        self.lr = args.lr
        self.schd = args.schd
        self.if_mask = args.if_mask
        self.log_batch = args.batch_to_wandb
        assert args.experiment in ['ns_labels','gt_labels'], 'Experiment label type is wrong!'
        self.experiment = args.experiment
        self.gt_avail = args.gt_avail
        self.cal_tr = args.cal_tr
        self.current_device = args.current_device
        self.display_inv = args.display_inv
        self.cls_dict = args.cls_dict
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()
        
        # for recording confidence values at each batch in on_train_batch_end
        self.training_step_confs = [torch.zeros((2,self.n_classes), dtype=float) for _ in range(2)]
        # for recording tp, fp, fn, tn in on_validation_epoch_end
        if not self.batch_wise: 
            self.validation_step_outputs = {}
            for k in ['s1', 's2']:
                self.validation_step_outputs[k] = {'tp': [], 'fp': [], 'fn': [], 'tn': []}
    
     
    # 2. forward propagation
    def forward(self, x1, x2):
        outs = self.net(x1, x2)
        return outs
    
    
    # 3. Optimizer
    def configure_optimizers(self):
        # optimizer
        if self.opt=='adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-8)
            print('Adam is used as optimizer!')
        elif self.opt=='sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
            print('SGD is used as optimizer!')
        else:
            raise ValueError("Please provide correct optimizer type (adam or sgd)!")
        # resume optimizer state
        if self.opt_w is not None:
            try:
                optimizer.load_state_dict(self.opt_w)
            except:
                raise TypeError("The type of optimizer stored in ckpt file is not compitable to the used optimizer!")
            # # send optimizer tensor to gpu
            # if args.opt=='adam': send_optim_tensor_gpu(optimizer,device)
            print(f"Training is resumed at epoch {self.last_epoch} from '{self.last_epoch}'!")
        # scheduler
        if self.schd>0: 
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.schd, factor=0.5)  
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "v_loss"}
        else:
            return optimizer
    
    
    # 4. training step
    def training_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        x1, x2 = x[:,:self.n_channels_1], x[:,self.n_channels_1:]
        del x
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x1, x2)
        if self.if_mask: 
            discard_masks = batch['mask_discard'].squeeze(axis=1)
            y[discard_masks!=0] = self.n_classes
        # calculate losses
        if self.sc:
            loss, conf_list = self._calculate_loss(pred_logits, y, train=True)
            self.training_step_confs[0] += conf_list
            self.training_step_confs[1] += (conf_list>0)
        else:
            loss = self._calculate_loss(pred_logits, y, train=True)
        # training accuracy
        if self.cal_tr:
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y, label_type=f'tr{pi+1}')
                if self.experiment == 'ns_labels' and self.gt_avail:
                    self._get_batch_acc(pred_logits_, y_gt, label_type=f'trg{pi+1}') 
        # training image examples - only log s2 imgs
        if batch_idx==0 and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('tr', x2, pred_logits[1], y_gt=y_gt)
            else:
                self._log_imgs('tr', x2, pred_logits[1], y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
        return loss
    
    
    def on_train_epoch_end(self):
        # get median confidence
        conf_tensor = self.training_step_confs[0]/self.training_step_confs[1]
        for ti, img_t in enumerate(['s1', 's2']):
            for c in range(self.n_classes):
                conf_c = conf_tensor[ti,c]
                self.log(f'conf_{img_t}_{c}', conf_c, sync_dist=True)
        # reinitialize the confidence list
        self.training_step_confs = [torch.zeros((2,self.n_classes), dtype=float) for _ in range(2)]
    
    
    # 5. validation step
    def validation_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        x1, x2 = x[:,:self.n_channels_1], x[:,self.n_channels_1:]
        del x
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            if self.gt_avail: y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_ns
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            y = y_gt
        # prediction
        pred_logits = self.forward(x1, x2)
        if self.batch_wise:
            # accuracy
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y, label_type=f'v{pi+1}')
                if self.experiment == 'ns_labels' and self.gt_avail:
                    self._get_batch_acc(pred_logits_, y_gt, label_type=f'vg{pi+1}')
        else:
            # confusion matrix
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_conf_matrix(pred_logits, y, key=f's{pi+1}')
        
        # validation loss
        loss = self._calculate_loss(pred_logits, y, train=False)
        self.log('v_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        # log images
        if batch_idx in [0,5,10] and self.current_device==0 and self.current_epoch%self.display_inv==0:
            if self.experiment=='gt_labels':
                self._log_imgs('v', x2, pred_logits[1], y_gt=y_gt)
            else:
                self._log_imgs('v', x2, pred_logits[1], y_ns=y_ns, y_gt=y_gt if self.gt_avail else None)
        del x1, x2, y, pred_logits
    
    
    def on_validation_epoch_end(self):
        if self.batch_wise:
            pass
        else:
            # summarize confusion matrix
            stats = {}
            for k in self.validation_step_outputs:
                stats[k] = {}
                for sk in self.validation_step_outputs[k]:
                    stats[k][sk] = torch.cat(self.validation_step_outputs[k][sk], dim=0)
                    stats[k][sk] = torch.sum(stats[k][sk], dim=0, keepdim=True)
                # calculate metrics
                self._get_batch_acc_each_epoch(stats[k], label_type='v'+k[1])
            # free memory
            for k in self.validation_step_outputs:
                for sk in self.validation_step_outputs[k]:
                    self.validation_step_outputs[k][sk].clear()  # free memory
    
    
    # 6. test step
    def test_step(self, batch, batch_idx):
        x = batch[KEY_IMG]
        x1, x2 = x[:,:self.n_channels_1], x[:,self.n_channels_1:]
        del x
        pred_logits = self.forward(x1, x2)
        if self.experiment=='ns_labels': 
            y_ns = batch[KEY_NS].squeeze(axis=1)
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y_ns, label_type=f'tsn{pi+1}')
                if self.gt_avail: 
                    y_gt = batch[KEY_GT].squeeze(axis=1)
                    self._get_batch_acc(pred_logits_, y_gt, label_type=f'ts{pi+1}')
        else:
            y_gt = batch[KEY_GT].squeeze(axis=1)
            for pi, pred_logits_ in enumerate(pred_logits):
                self._get_batch_acc(pred_logits_, y_gt, label_type=f'ts{pi+1}')
        del x1, x2, pred_logits
        
    
    # auxiliary functions
    # a.1. get model
    def _get_model(self, args):
        if args.resume in ['imagenet','ssl','swsl']: assert args.n_channels==3, \
            'Pretrained encoder weights in smp package only support 3 bands as input!'
        
        if args.fuse_type=='late':
            structure = Unet_lateF_3branch if self.use_com_feat else Unet_lateF_2branch
        elif args.fuse_type=='mid':
            structure = Unet_midF_3branch if self.use_com_feat else Unet_midF_2branch
        mtag = '3' if self.use_com_feat else '2'
        self.net = structure(encoder_name=args.mt,           # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                             encoder_weights=args.resume if args.resume in ['imagenet','ssl','swsl'] else None,  # pre-trained weights for encoder initialization
                             in_channels_1=self.n_channels_1,    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                             in_channels_2=self.n_channels_2,    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                             classes=self.n_classes,         # model output channels (number of classes in your dataset)
                            )
        print(f"Construct {mtag} encoder branch UNet model with {args.mt} as backbone!")
        
        # load weights from a checkpoint file - tbd
        self.opt_w = None
        if args.resume: 
            if args.resume in ['imagenet','ssl','swsl']:
                print(f"Pre-trained model is loaded from smp package (type:{args.resume})!")
            else:
                assert os.path.isfile(args.resume), 'Given model checkpoint loading path is invalid!'
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
                msd = modify_module_state_dict_keys(checkpoint['model_state_dict'])
                self.net.load_state_dict(msd)
                # initialize or resume
                if args.prt:
                    print(f"Pre-trained model is loaded from '{args.resume}'!")
                else:
                    self.opt_w = checkpoint['optimizer_state_dict']
                    self.last_epoch = checkpoint['epoch']
        else:
            print('No pretrained weights, will start from scratch.')
    
    
    # a.2. get losses
    def _get_loss(self, n_classes, loss_type, fuse_type, if_mask=False, label_smoothing=False):
        # - segmentation losses
        self.losses_seg = {}
        # Cross Entropy
        if 'c' in loss_type:
            if n_classes>1:
                if if_mask:
                    self.CELoss = nn.CrossEntropyLoss(ignore_index=n_classes)  
                else:
                    self.CELoss = nn.CrossEntropyLoss()
                print("CrossEntropy loss is used!")
            else:
                self.CELoss = nn.BCEWithLogitsLoss()
                print("BinaryCrossEntropy loss is used!")
            self.losses_seg['c'] = self.CELoss
        else:
            self.CELoss = None  
        # Dice
        if 'd' in loss_type:
            if label_smoothing:
                def soft_dice_loss(y_pred, y_true):
                    if self.n_classes>1:
                        y_pred = y_pred.log_softmax(dim=1).exp()
                    else:
                        y_pred = F.logsigmoid(y_pred).exp()
                    scores = soft_dice_score(y_pred, y_true.type_as(y_pred), dims=(0,2,3))  # class-wise
                    loss = 1.0 - scores
                    return loss.mean()
                self.DLoss = soft_dice_loss
            else:
                self.DLoss = DiceLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                    from_logits=True, 
                                    ignore_index=n_classes if if_mask else None)
            print("Dice loss is used!")
            self.losses_seg['d'] = self.DLoss
        else:
            self.DLoss = None
        # Focal loss
        if 'f' in loss_type:
            self.FLoss = FocalLoss(mode='multiclass' if n_classes>1 else 'binary', 
                                   ignore_index=n_classes if if_mask else None)
            print("Focal loss is used!")
            self.losses_seg['f'] = self.FLoss
        else:
            self.FLoss = None
        assert len(self.losses_seg)>0, 'No segmentation loss is used!'
        
        # - consistency losses
        if self.cl_type == 'mse':
            # MSE loss
            self.CONLoss = nn.MSELoss(reduction='none')  # If reduction is ‘none’, then the same shape as the input: (N,C,H,W)
        elif self.cl_type == 'kl':
            # KL divergence
            self.CONLoss = nn.KLDivLoss(reduction='none', log_target=True)  # If reduction is ‘none’, then the same shape as the input: (N,C,H,W)
        elif self.cl_type == 'ce':
            if n_classes>1:
                self.CONLoss = nn.CrossEntropyLoss(reduction='none')  # If reduction is ‘none’, (N,H,W)
            else:
                self.CONLoss = nn.BCEWithLogitsLoss(reduction='none')  
        elif self.cl_type == 'none':
            assert fuse_type=='mid', f'Consistency loss type is required when the fusion type is {fuse_type}!'
            print('No consistency loss is used!')
        
        
    # a.3. get batch accuracy
    def _get_batch_acc(self, pred_logits, y, label_type='tr'):
        log_batch = self.log_batch if 'tr' in label_type else False
        metrics = evl.evaluate_batch_pl(pred_logits, y, self.n_classes)
        for mk in metrics:
            metr = metrics[mk]
            if torch.prod(torch.tensor(metr.size()))>1:
                for i in range(len(metr)-1):
                    self.log(f'{label_type}_{mk}_{i}', metr[i], on_epoch=True, on_step=log_batch, sync_dist=True)
                self.log(f'{label_type}_m{mk}', metr[-1], on_epoch=True, on_step=log_batch, sync_dist=True)
            else:
                self.log(f'{label_type}_{mk}', metr, on_epoch=True, on_step=log_batch, sync_dist=True)
    
    # a.3-2. get accuracy from confusion matrix after every epoch
    def _get_batch_conf_matrix(self, pred_logits, y, key):
        metrics = evl.get_confuse_matrix_batch_pl(pred_logits, y, self.n_classes)
        for mk in metrics:
            self.validation_step_outputs[key][mk].append(metrics[mk])
    
    def _get_batch_acc_each_epoch(self, stats, label_type='v'):
        metrics = evl.get_batch_acc_from_confuse_matrix(stats, self.n_classes)
        for mk in metrics:
            metr = metrics[mk]
            if torch.prod(torch.tensor(metr.size()))>1:
                for i in range(len(metr)-1):
                    self.log(f'{label_type}_{mk}_{i}', metr[i], sync_dist=True)
                self.log(f'{label_type}_m{mk}', metr[-1], sync_dist=True)
            else:
                self.log(f'{label_type}_{mk}', metr, sync_dist=True)
                
    # a.4. log images
    def _log_imgs(self, prefix, x, pred_logits, y_gt=None, y_ns=None):
        mask_data = pred_logits[0].detach().cpu().numpy().argmax(axis=0)
        class_labels = self.cls_dict
        masks={'prediction': {'mask_data': mask_data, 'class_labels': class_labels},}
        if y_gt is not None:
            masks['ground truth'] = {'mask_data': y_gt[0].detach().cpu().numpy(), 'class_labels': class_labels}
        if y_ns is not None:
            masks['noisy labels'] = {'mask_data': y_ns[0].detach().cpu().numpy(), 'class_labels': class_labels}
        # get rgb img
        if x.shape[1]==3:
            rgb_img = x[0].detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[1]>3:
            rgb_img = x[0,[3,2,1]].detach().cpu().numpy().transpose((1,2,0))
        elif x.shape[1]<3:
            rgb_img = x[0,0].detach().cpu().numpy()
        mask_img = wandb.Image(rgb_img, masks=masks)
        wandb.log({f'{prefix}_masks': mask_img})
        
        
    # a.4. claculate loss
    def _calculate_loss(self, pred_logits, y, train=True):
        # label smoothing
        if self.ls:
            # convert y to one-hot
            y = F.one_hot(y, num_classes=self.n_classes).float().permute((0,3,1,2))
            if train:
                y = self._get_smoothed_labels(y)
        # calculate losses
        loss = torch.tensor(0.0, device=self.device)
        # segmentation losses
        for pi, pred_logits_ in enumerate(pred_logits):
            loss_branch = torch.tensor(0.0, device=self.device)
            for lk in self.losses_seg:
                L = self.losses_seg[lk]
                loss_ = L(pred_logits_, y)
                loss_branch += loss_
            if train: self.log(f'tr_segloss_{pi+1}', loss_branch, on_epoch=True, on_step=self.log_batch, sync_dist=True)
            loss += loss_branch
        if train: self.log('tr_segloss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        # consistency losses
        if self.cl_type != 'none':
            selection = self.sc if train else False
            if selection:
                loss_con, conf_list = self._calculate_con_loss(pred_logits, train, selection=selection)
            else:
                loss_con = self._calculate_con_loss(pred_logits, train, selection=selection)
            loss += self.cl_weight*loss_con
            if train: self.log('tr_conloss', loss_con, on_epoch=True, on_step=self.log_batch, sync_dist=True)
            if train: self.log('tr_loss', loss, on_epoch=True, on_step=self.log_batch, sync_dist=True)
            if selection:
                return loss, conf_list
        return loss
    
    def _calculate_con_loss(self, pred_logits, train, selection=False):
        # number of samples
        if selection:
            sel_masks = []
            conf_list = []
            for ref_logits in pred_logits:
                sel_mask_, conf_list_ = self._get_selection_mask_by_confidence(ref_logits)
                sel_masks.append(sel_mask_)
                conf_list.append(conf_list_)
            n_sam1, n_sam2 = [sel_mask.sum() for sel_mask in sel_masks]
        else:
            n_sam1 = n_sam2 = pred_logits[0].numel()/self.n_classes
        # calculate pixel-wise consistency loss
        if not self.use_com_feat:
            if self.cl_type == 'kl':
                pred_logs = [F.log_softmax(pred_logits_, dim=1) for pred_logits_ in pred_logits]
                loss_con1_ = self.CONLoss(pred_logs[0], pred_logs[1].detach())
                loss_con2_ = self.CONLoss(pred_logs[1], pred_logs[0].detach())
            elif self.cl_type == 'ce':
                loss_con1_ = self.CONLoss(pred_logits[0], F.softmax(pred_logits[1].detach(), dim=1))
                loss_con2_ = self.CONLoss(pred_logits[1], F.softmax(pred_logits[0].detach(), dim=1))
            elif self.cl_type == 'mse':
                pred_probs = [F.softmax(pred_logits_, dim=1) for pred_logits_ in pred_logits]
                loss_con1_ = self.CONLoss(pred_probs[0], pred_probs[1].detach())
                loss_con2_ = self.CONLoss(pred_probs[1], pred_probs[0].detach())
        # summation
        if selection:
            loss_con1_ = loss_con1_*sel_masks[1]
            loss_con2_ = loss_con2_*sel_masks[0]
        loss_con1 = loss_con1_.sum()/n_sam2
        loss_con2 = loss_con2_.sum()/n_sam1
        if train: self.log('tr_conloss_2t', loss_con1, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        if train: self.log('tr_conloss_2i', loss_con2, on_epoch=True, on_step=self.log_batch, sync_dist=True)
        loss_con = loss_con1 + loss_con2
        
        if selection:
            return loss_con, torch.tensor(conf_list)
        else:
            return loss_con
    
    
    # a.5. label smoothing
    def _get_smoothed_labels(self, y):
        # smooth labels
        y = y*(1-self.lsf) + self.lsf/self.n_classes
        return y

    # a.6. get sample selection mask according to confidence
    def _get_selection_mask_by_confidence(self, ref_logits):
        ref_sfmx = F.softmax(ref_logits, dim=1)
        ref_conf, ref_lbls = torch.max(ref_sfmx, dim=1)
        sc_ratio = self.sc_rmup_func(self.current_epoch, self.sc_rmup_len)*self.sc_conf_prop
        select_locs = ()
        conf_list = []
        # 1. select samples from each class to calculate consistency loss
        for c in range(self.n_classes):
            ref_lbls_c = ref_lbls==c
            ref_locs = ref_lbls_c.nonzero(as_tuple=True)
            ref_conf_c = ref_conf[ref_locs]
            n_sam = len(ref_conf_c)
            # if no samples in this class, skip
            if n_sam==0: 
                conf_list.append(0)
                continue
            # sample selection number
            n_csam = int(n_sam*sc_ratio)
            n_usam = int((n_sam-n_csam)*self.sc_unc_prop)
            # selecting confident samples
            if n_csam>0:
                out = torch.kthvalue(-ref_conf_c, n_csam) # Returns a namedtuple (values, indices) where values is the k th smallest element of each row of the input tensor in the given dimension dim
                conf_thres = -out.values
                conf_locs = tuple([ref_locs_[ref_conf_c>=conf_thres] for ref_locs_ in ref_locs])
                unc_locs_cand = tuple([ref_locs_[ref_conf_c<conf_thres] for ref_locs_ in ref_locs])
                if select_locs:
                    select_locs = tuple([torch.cat((select_locs[i], conf_locs[i])) for i in range(len(select_locs))])
                else:
                    select_locs = conf_locs
            else:
                unc_locs_cand = ref_locs
                conf_thres = 0
            conf_list.append(conf_thres)
            # selecting uncertain samples
            if n_usam>0:
                rand_inds = torch.randperm(len(unc_locs_cand[0]))[:n_usam]
                unc_locs = tuple([unc_locs_cand_[rand_inds] for unc_locs_cand_ in unc_locs_cand])
                if select_locs:
                    select_locs = tuple([torch.cat((select_locs[i], unc_locs[i])) for i in range(len(select_locs))])
                else:
                    select_locs = unc_locs
        # 2. get selection mask
        select_mask = ref_lbls*0
        select_mask[select_locs] = 1
        del ref_conf, ref_lbls
        
        return select_mask, conf_list
        
        
            
                
                
            
    