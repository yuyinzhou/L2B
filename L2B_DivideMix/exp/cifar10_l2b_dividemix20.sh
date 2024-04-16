#!/bin/bash

source activate lxh_l2b

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install  scikit-learn higher torchnet

CUDA_VISIBLE_DEVICES=4 python3 Train_cifar.py --need_clean=False  --single_meta 1 --seed=0 --r=0.2 --lambda_u=0  --log_wandb=True --wandb_project='L2B_extend' --wandb_experiment='l2b_no_clean_dividemix_cifar10_repro_nr20_lr002_batch64_seed0'


#CUDA_VISIBLE_DEVICES=0 python3 Train_cifar.py --single_meta 1 --r  0.5 --lr 0.10