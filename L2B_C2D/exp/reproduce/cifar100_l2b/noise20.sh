#!/bin/bash

source activate lxh_c2d

pip install wandb


CUDA_VISIBLE_DEVICES=6 python3 ~/code/c2d/main_cifar.py --r 0.2 --lambda_u 25 \
--dataset cifar100 --p_threshold 0.03 --data_path cifar-100/cifar-100-python  --experiment-name simclr_resnet18 \
 --method selfsup --net resnet18   --use_l2b \
 --log_wandb=True --wandb_project='C2D_extend' --wandb_experiment='C2D_baseline_noisy_l2b_cifar100_noise20'