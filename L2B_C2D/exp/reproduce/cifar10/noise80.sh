#!/bin/bash

export PATH=/usr/local/anaconda3/bin/:$PATH && source activate lxh_c2d
#pip install torch==1.13.1+cu116 torchvision torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
#pip3 install  scikit-learn higher torchnet
#
#pip install wandb
CUDA_VISIBLE_DEVICES=2 python3 ~/code/c2d/main_cifar.py --r 0.8 --lambda_u 25 \
--dataset cifar10 --p_threshold 0.03 --data_path cifar-10/cifar-10-batches-py/  --experiment-name simclr_resnet18 \
 --method selfsup --net resnet18 \
 --log_wandb=True --wandb_project='C2D_extend' --wandb_experiment='C2D_baseline_cifar10_noise80'