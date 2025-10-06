import datetime
import os
import sys
import numpy as np

# appending self-defined package path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from torch.utils.data import DataLoader, ConcatDataset

# import lightening
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from sam2.get_args import get_args_train_baseline as get_args
from sam2.sam2_image_trainer import SAM2ImageTrainer
from sam2.sam2_image_w_auxDe_trainer import SAM2ImageTrainerAux
from sam2.sam2_image_w_sepAuxDe4Weak_trainer import SAM2ImageTrainerSepAuxDe4Weak

# dataset
from dataset.get_combined_datasets import load_combined_datasets, get_datasets_train, MyDistributedBatchSampler
from dataset.datasets_settings import datasets_settings_test
from dataset import DWOSMDataset, OpenEarthMapDataset
import albumentations as A


INPUT_SIZE = 256

# # # # # # # # # # main function # # # # # # # # # # 
def main():
    t0 = datetime.datetime.now().replace(microsecond=0)

    ###### 1 - parameter setting ######
    global args
    args = get_args()
    # resume option
    if args.resume:
        ckpt_path = args.resume
        wandb_resume = False
    else:
        ckpt_path = None
        wandb_resume = False
    # Setting all the random seeds to the same value.
    # This is important in a distributed training setting. 
    # Each rank will get its own set of initial weights. 
    # If they don't match up, the gradients will not match either,
    # leading to training that may not converge.
    pl.seed_everything(args.seed)
    # setting logger
    os.environ["WANDB_MODE"] = args.wmode
    wandb_logger = WandbLogger(project=args.pjn, entity=args.entity, 
                               group=args.gname, job_type=args.job_type,
                               name=f's{args.seed}_b{args.bs}',
                               save_dir=args.pdir,
                               resume=wandb_resume) 
    
    ###### 2 - load data ######
    # datasets
    trans_tr = A.Compose([A.RandomCrop(width=args.patch_size, height=args.patch_size),
                        #   A.Resize(width=INPUT_SIZE, height=INPUT_SIZE, interpolation=cv2.INTER_NEAREST),
                          A.D4(p=0.5), # A.Flip(p=0.5),
                          A.Rotate(p=0.2),
                          ])
    trans_ts = A.Compose([A.CenterCrop(width=args.patch_size, height=args.patch_size, )])

    # val
    val_dataset = DWOSMDataset(datasets_settings_test['dw']['data_path'],
                               mode=datasets_settings_test['dw']['mode'],
                               n_bands=datasets_settings_test['dw']['n_channels'],
                               transform=trans_ts)
    subset_inds = np.arange(9).tolist()+[20]
    val_dataset = torch.utils.data.Subset(val_dataset, subset_inds)  # range(args.tbs))
    val_loader = DataLoader(val_dataset, batch_size=args.tbs, num_workers=args.num_workers,
                            pin_memory=True, shuffle=False, drop_last=True)
    # # test
    # test_dataset = OpenEarthMapDataset(datasets_settings_test['openearthmap']['data_path'],
    #                                    transform=SegAugTransforms(trans_ts))
    # test_loader = DataLoader(test_dataset, batch_size=args.tbs, num_workers=args.num_workers,
    #                         pin_memory=True, shuffle=False, drop_last=True)
    # train
    if args.ntasks > 1:
        datasets_tr = get_datasets_train(args.dataset_name, transform=trans_tr, 
                                         subset_proportion=args.subset_proportion, seed=args.seed)
        class MyDataModule(pl.LightningDataModule):
            def __init__(self, max_dataset_size, num_workers, batch_size=32):
                super().__init__()
                # self.dataset_names = dataset_names
                self.max_dataset_size = max_dataset_size
                self.batch_size = batch_size
                self.num_workers = num_workers
                # self.transform = transform

            def setup(self, stage):
                self.datasets, self.split_rates = datasets_tr # get_datasets_train(self.dataset_names, self.transform)

            def train_dataloader(self):
                sampler = MyDistributedBatchSampler(self.datasets, self.max_dataset_size, self.split_rates,
                                                    batch_size=self.batch_size, 
                                                    shuffle=True, drop_last=True, seed=args.seed,
                                                    num_replicas=self.trainer.world_size, rank=self.trainer.global_rank)
                dataset_all = ConcatDataset(self.datasets)
                return DataLoader(dataset_all, batch_sampler=sampler,
                                  num_workers=self.num_workers, pin_memory=True)
            
            def val_dataloader(self):
                return val_loader
        # pl data module
        train_loader = MyDataModule(max_dataset_size=args.max_dataset_size, batch_size=args.bs, num_workers=args.num_workers)
        # batch number
        if args.max_dataset_size is None:
            batch_nums = [len(ds)//args.bs for ds in datasets_tr[0]]
        else:
            batch_nums = [min(len(ds), args.max_dataset_size)//args.bs for ds in datasets_tr[0]]
        total_batch_num = sum(batch_nums)
        batch_num = (total_batch_num // args.ntasks) * (args.ntasks)
        # print(batch_nums, total_batch_num, args.ntasks, batch_num)
    else:
        train_loader, batch_num = load_combined_datasets(args.dataset_name, args.bs, args.num_workers,
                                                         transform=trans_tr, shuffle=True, drop_last=True,
                                                         subset_proportion=args.subset_proportion,
                                                         max_dataset_size=args.max_dataset_size)

    # batch number    
    args.n_batch_tr = batch_num
    print(f'Total number of training batches: {batch_num} with batch size of {args.bs}.')


    ###### 3 - define the model ######
    torch.set_float32_matmul_precision('high')
    if args.use_sep_aux_de_weak:
        model = SAM2ImageTrainerSepAuxDe4Weak(args)
    elif args.use_aux_decoder:
        model = SAM2ImageTrainerAux(args)
    else:
        model = SAM2ImageTrainer(args)
    
    
    ###### 4 - train the model ######
    print('\nStart training...')
    callbacks = [ModelCheckpoint(dirpath = args.checkpoints_dir, every_n_epochs = args.save_inv, 
                                #  save_weights_only=True, filename='model_{epoch:02d}_{val_loss:.2f}',
                                 save_top_k=-1,  # Save all models
                                 save_last=False,  # Do not save the last model separately 
                                 )]
    if args.schdt!='none': callbacks += [LearningRateMonitor(logging_interval='epoch')]
    trainer = pl.Trainer(accelerator=args.accelerator,
                         devices=args.devices,
                         strategy=args.strategy,
                         num_nodes=args.nodes,
                         precision='16-mixed' if args.amp else '32-true',
                         max_epochs=args.epochs, 
                         check_val_every_n_epoch=args.display_inv,
                         logger=wandb_logger,
                         enable_progress_bar=True,
                         callbacks=callbacks,
                         use_distributed_sampler=(args.ntasks <= 1),
                         )    
    
    if args.ntasks > 1:
        trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # # test the model
    # trainer.test(model, dataloaders=test_loader, verbose=True)
    
    t1 = datetime.datetime.now().replace(microsecond=0)
    print(f'Training is finished|Total spent time:{t1-t0}!')
    
    
    
if __name__ == '__main__':
    main()


