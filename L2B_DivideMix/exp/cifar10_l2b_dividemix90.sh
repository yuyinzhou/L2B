#!/bin/bash
source activate lxh_nnunet
#pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install  scikit-learn higher torchnet wandb

CUDA_VISIBLE_DEVICES=2 python3 Train_cifar.py --single_meta 1 --seed=0 --need_clean=True --r=0.9 --p_threshold=0.6  --lambda_u=50  --log_wandb=True --wandb_project='L2B_extend' --wandb_experiment='l2b_yes_1000_clean_cifar10_repro_nr90_lr002_batch64_seed0'


#CUDA_VISIBLE_DEVICES=0 python3 Train_cifar.py --single_meta 1 --r  0.5 --lr 0.10