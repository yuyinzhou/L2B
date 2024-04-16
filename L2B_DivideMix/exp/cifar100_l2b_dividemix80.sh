#!/bin/bash
source activate lxh_nnunet

CUDA_VISIBLE_DEVICES=4 python3 Train_cifar100.py --need_clean=True --single_meta 1 --seed=0 --r  0.8 --lambda_u=150 --log_wandb=True --wandb_project='L2B_extend' --wandb_experiment='l2b_yes_clean_1000_cifar100_dividemix_nr80_lr002_batch64_seed0'


#CUDA_VISIBLE_DEVICES=0 python3 Train_cifar.py --single_meta 1 --r  0.5 --lr 0.10