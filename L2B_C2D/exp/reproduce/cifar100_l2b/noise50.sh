#!/bin/bash

source activate nnunet

CUDA_VISIBLE_DEVICES=7 python3 ~/code/c2d/main_cifar.py --r 0.5 --lambda_u 150 \
--dataset cifar100 --p_threshold 0.03 --data_path cifar-100/cifar-100-python  --experiment-name simclr_resnet18 \
 --method selfsup --net resnet18  --num_meta 1000 --need_clean   --use_l2b \
 --log_wandb=True --wandb_project='C2D_extend' --wandb_experiment='C2D_baseline_l2b_clean_1000_cifar100_noise50'