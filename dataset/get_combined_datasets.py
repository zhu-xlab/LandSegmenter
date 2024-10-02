import cv2
import math
import random
import numpy as np

import torch
from torch.utils.data import Sampler, DistributedSampler, DataLoader, ConcatDataset

import pytorch_lightning as pl
import albumentations as A

from dataset import DWOSMDataset, DyEarthDataset, OpenEarthMapDataset, SegAugTransforms
from dataset.datasets_settings import datasets_settings_train, datasets_settings_test

def chunk(indices, size):
    return torch.split(torch.tensor(indices), size)

# Define a custom batch sampler for single node training
class MyBatchSampler(Sampler):
    def __init__(self, datasets, batch_sizes, shuffle=True, drop_last=True, max_dataset_size=None):
        self.n_datasets = len(datasets)
        if isinstance(batch_sizes, int):
            self.batch_sizes = [batch_sizes] * len(datasets)
        elif len(batch_sizes) == 1:
            self.batch_sizes = batch_sizes * len(datasets)
        else:
            self.batch_sizes = batch_sizes
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.max_dataset_size = max_dataset_size

        # Get the number of images and overall indices for each dataset
        self.n_imgs = [len(dataset) for dataset in datasets]
        self.indices = [list(range(sum(self.n_imgs[:i]), sum(self.n_imgs[:i])+ n_img)) \
                        for i, n_img in enumerate(self.n_imgs)]

    def __iter__(self):
        # Randomly shuffle the indices within each subset
        if self.shuffle:
            for i in range(self.n_datasets):
                random.shuffle(self.indices[i])
        
        # chunk the indices
        batches = []
        for i in range(self.n_datasets):
            if self.max_dataset_size is None:
                b_ = chunk(self.indices[i], self.batch_sizes[i])
            else:
                b_ = chunk(self.indices[i][:self.max_dataset_size], self.batch_sizes[i])
            if self.drop_last:
                if len(self.indices[i]) % self.batch_sizes[i] != 0:
                    b_ = b_[:-1]
            batches += b_
        
        # all the batches
        batches = [batch.tolist() for batch in batches]
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.max_dataset_size is None:
            n_imgs = self.n_imgs
        else:
            assert self.max_dataset_size >= max(self.batch_sizes), \
                    f'Maximum dataset size {self.max_dataset_size} is smaller than max batch size {max(self.batch_sizes)}.'
            n_imgs = [min(n_img, self.max_dataset_size) for n_img in self.n_imgs]

        if self.drop_last:
            n_batches = [n_img // batch_size for n_img, batch_size in zip(n_imgs, self.batch_sizes)]
        else:
            n_batches = [math.ceil(n_img / batch_size) for n_img, batch_size in zip(n_imgs, self.batch_sizes)]
        return sum(n_batches)
    

# Define a custom batch sampler for distributed training
class MyDistributedBatchSampler(DistributedSampler):
    def __init__(self, datasets, max_dataset_size,
                 num_replicas=None,
                 rank=None, shuffle=True,
                 seed = 0, drop_last = False, batch_size = 10):
        dataset  = ConcatDataset(datasets)
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.n_datasets = len(datasets)
        self.batch_size = batch_size
        self.max_dataset_size = max_dataset_size

        # Get the number of images and overall indices for each dataset
        self.n_imgs = [len(dataset) for dataset in datasets]
        self.indices = [list(range(sum(self.n_imgs[:i]), sum(self.n_imgs[:i])+ n_img)) \
                        for i, n_img in enumerate(self.n_imgs)]
        
        # number of batches
        num_batches_list = []
        for indice in self.indices:
            n_inds = min(len(indice), self.max_dataset_size) if self.max_dataset_size is not None else len(indice)
            if self.drop_last:
                num_batches_list.append(n_inds // self.batch_size)
            else:
                num_batches_list.append(math.ceil(n_inds/self.batch_size))

        num_batches = sum(num_batches_list)

        self.num_batches = math.floor(num_batches / self.num_replicas)
        self.total_size = self.num_batches * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            for i in range(self.n_datasets):
                random.shuffle(self.indices[i])

        # chunk the indices
        batches = []
        for i in range(self.n_datasets):
            if self.max_dataset_size is None:
                b_ = chunk(self.indices[i], self.batch_size)
            else:
                b_ = chunk(self.indices[i][:self.max_dataset_size], self.batch_size)
            if self.drop_last:
                if len(self.indices[i]) % self.batch_size != 0:
                    b_ = b_[:-1]
            batches += b_
        
        # all the batches
        batches = [batch.tolist() for batch in batches]
        if self.shuffle:
            random.shuffle(batches)

        batches = batches[:self.total_size]
        assert len(batches) == self.total_size

        rank_batches = batches[self.rank:len(batches):self.num_replicas]
        return iter(rank_batches)

    def __len__(self) -> int:
        return self.num_batches

# load training datasets
def get_datasets_train(dataset_names, transform=None, subset_proportion=1, seed=0):
    if subset_proportion < 1: np.random.seed(seed)
    trans = transform if transform is None else SegAugTransforms(transform)
    datasets_settings = datasets_settings_train
    
    # load datasets
    datasets = []
    for dname in dataset_names:
        if dname == 'dw':
            ds_args = datasets_settings['dw']
            dataset = DWOSMDataset(ds_args['data_path'], 
                                   mode=ds_args['mode'], 
                                   n_bands=ds_args['n_channels'], 
                                   transform=trans)
            if subset_proportion < 1:
                dataset = get_subset_datasets_train(dataset, subset_proportion)
            datasets.append(dataset)
        elif dname == 'dynamicearthnet':
            ds_args = datasets_settings['dynamicearthnet']
            dataset = DyEarthDataset(ds_args['data_path'],
                                    transform=trans)
            if subset_proportion < 1:
                dataset = get_subset_datasets_train(dataset, subset_proportion)
            datasets.append(dataset)
        elif dname == 'openearthmap':
            ds_args = datasets_settings['openearthmap']
            dataset = OpenEarthMapDataset(ds_args['data_path'],
                                          transform=trans)
            if subset_proportion < 1:
                dataset = get_subset_datasets_train(dataset, subset_proportion)
            datasets.append(dataset)
        else:
            raise ValueError(f'Invalid dataset name: {dname}')
        # print(f'Loaded {len(dataset)} samples from {dname}.')
    
    return datasets

def get_subset_datasets_train(dataset, subset_proportion=1):
    n_samples = int(len(dataset) * subset_proportion)
    dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), n_samples, replace=False))
    return dataset

# data loader for single node training
def load_combined_datasets(dataset_names, batch_size, num_workers, transform=None, 
                           shuffle=True, drop_last=True, max_dataset_size=None):
    # load datasets
    datasets = get_datasets_train(dataset_names, transform=transform)
    dataset_all = ConcatDataset(datasets)

    # sampler
    batch_sampler = MyBatchSampler(datasets, batch_size, 
                                    shuffle=shuffle, drop_last=drop_last, 
                                    max_dataset_size=max_dataset_size)
    # data loader
    data_loader = DataLoader(
        dataset_all,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    batch_num = len(batch_sampler)

    return data_loader, batch_num


if __name__ == '__main__':
    patch_size = 256
    dataset_names = ['dw', 'dynamicearthnet', 'openearthmap']
    batch_size = 6
    num_workers = 4

    data_loader = load_combined_datasets(patch_size, dataset_names, batch_size, num_workers,
                                max_dataset_size=1000)
    print(f'Loaded {len(data_loader)} batches.')

    for i, batch in enumerate(data_loader):
        print(f'Batch {i}: {batch[0].shape}, {batch[1].shape}')
        if i == 1:
            break