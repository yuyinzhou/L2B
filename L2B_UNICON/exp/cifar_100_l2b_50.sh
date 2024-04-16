


source activate nnunet

CUDA_VISIBLE_DEVICES=1 python Train_cifar.py --need_clean=True --log_wandb=True --wandb_project='L2B_extend' --wandb_experiment='l2b_unicon_yes_clean_1000_cifar100_repro_nr50_lr002_batch64_seed0' --single_meta=1 --dataset cifar100 --num_class 100 --data_path ./data/cifar-100 --noise_mode 'sym' --r 0.5