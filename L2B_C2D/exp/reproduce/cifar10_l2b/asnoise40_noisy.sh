#!/bin/bash

export PATH=/usr/local/anaconda3/bin/:$PATH && source activate lxh_c2d


CUDA_VISIBLE_DEVICES=5 python3 ~/code/c2d/main_cifar.py --r 0.4 --lambda_u 25 --noise_mode asym \
--dataset cifar10 --p_threshold 0.03 --data_path  cifar-10/cifar-10-batches-py   --experiment-name simclr_resnet18 \
 --method selfsup --net resnet18  --num_meta 1000  --use_l2b \
 --log_wandb=True --wandb_project='C2D_extend' --wandb_experiment='C2D_baseline_noisy_asys_l2b_cifar10_noise40'