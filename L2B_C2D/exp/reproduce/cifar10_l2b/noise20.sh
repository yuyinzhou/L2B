#!/bin/bash

export PATH=/usr/local/anaconda3/bin/:$PATH && source activate lxh_c2d


CUDA_VISIBLE_DEVICES=3 python3 ~/code/c2d/main_cifar.py --r 0.2 --lambda_u 0 \
--dataset cifar10 --p_threshold 0.03 --data_path cifar-10/cifar-10-batches-py/  --experiment-name simclr_resnet18 \
 --method selfsup --net resnet18 --need_clean=True   --use_l2b \
 --log_wandb=True --wandb_project='C2D_extend' --wandb_experiment='C2D_baseline_noisy_l2b_cifar10_noise20'