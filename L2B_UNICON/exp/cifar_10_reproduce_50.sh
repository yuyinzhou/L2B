

source activate lxh_nnunet

CUDA_VISIBLE_DEVICES=6 python Train_cifar.py --dataset cifar10 --num_class 10 --data_path /home/lxh/code/l2b_dividemix/cifar-10 --noise_mode 'sym' --r 0.5