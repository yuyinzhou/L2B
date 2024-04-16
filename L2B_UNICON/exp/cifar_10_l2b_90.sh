

#export PATH=/usr/local/anaconda3/bin/:$PATH && source activate nnunet

pip3 install  scikit-learn higher torchnet wandb

CUDA_VISIBLE_DEVICES=0 python Train_cifar.py --need_clean=True --dataset cifar10 --single_meta=1 --num_class 10 --num_epochs=300 --data_path ./data/cifar-10 --noise_mode 'sym' --r 0.9 --lambda_u 50 --log_wandb=True --wandb_project='L2B_extend' --wandb_experiment='l2b_unicon_yes_clean_cifar90_repro_nr20_lr002_batch64_seed0'