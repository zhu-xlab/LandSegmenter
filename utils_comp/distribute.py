# -*- coding: utf-8 -*-
"""
Package for faciliting distribution computing

Created on Fri Jun 23 14:30:38 2023

@author: liu_ch
"""

import os
import torch

### distributed running ###
def init_distributed_mode(args):
    # 1> prepare parallel-related parameters by reading from environment variables 
    args.slurm = "SLURM_JOB_ID" in os.environ

    # rank (index of current process) and world size (total number of processes) settings
    if args.slurm:
        # for slurm
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"][0])
    else:
        # for local multi-GPU job - jobs started with torch.distributed.launch
        # read environment variables
        print('No slurm found!')
        args.rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])

    # 2> init the process in context
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank)

    # 3> get cuda device id
    print(f'n of GPUs: {torch.cuda.device_count()}')
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    
    return 

# def get_slurm_info(args):
#     args.slurm = "SLURM_JOB_ID" in os.environ
#     if args.slurm:
#         args.nodes = os.environ["SLURM_NNODES"]
#         args.devices = os.environ["SLURM_TASKS_PER_NODE"]
#         args.strategy = 'ddp'
#         print(f'Slurm found with {args.nodes} nodes, {args.devices} devices per node.')
#     else:
#         print('No slurm found!')
#     return args

# def get_local_device_info(args):
#     args.nodes = 1
#     # gpu
#     if args.accelerator == 'gpu':
#         args.devices = torch.cuda.device_count()
#         if args.devices == 0:
#             args.accelerator = 'cpu'
#             print('No GPU found, switch to CPU mode.')
#         else:
#             print(f'Uising {args.devices} GPUs for training!')
#     # cpu
#     if args.accelerator == 'cpu':
#         args.devices = 1
#         print('Uising CPU for training!')
#         # args.devices = os.cpu_count()
#         # print(f'Uising {args.devices} CPU cores for training!')

#     args.strategy = 'ddp' if args.devices>1 else 'auto'
#     return args
    
def get_device_info(args):
    args.slurm = "SLURM_JOB_ID" in os.environ
    if args.slurm:
        args.nodes = os.environ["SLURM_NNODES"]
        args.devices = os.environ["SLURM_TASKS_PER_NODE"]
        args.strategy = 'ddp'
        print(f'Slurm found with {args.nodes} nodes, {args.devices} devices per node.')
    else:    
        args.nodes = 1
        # gpu
        if args.accelerator == 'gpu':
            args.devices = torch.cuda.device_count()
            if args.devices == 0:
                args.accelerator = 'cpu'
                print('No GPU found, switch to CPU mode.')
            else:
                print(f'No slurm found! Uising {args.devices} local GPUs for training!')
        # cpu
        if args.accelerator == 'cpu':
            args.devices = 1
            print('Uising CPU for training!')
            # args.devices = os.cpu_count()
            # print(f'Uising {args.devices} CPU cores for training!')

    ndevs = args.nodes * args.devices
    args.strategy = 'ddp' if ndevs>1 else 'auto'
    return args
        

