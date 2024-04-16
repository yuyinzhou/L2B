

#source activate lxh_l2b
#pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
#pip3 install  scikit-learn higher torchnet

CUDA_VISIBLE_DEVICES=6 python Train_cifar.py --need_clean=True --dataset cifar10 --single_meta=1 --num_class 10 --num_epochs=300 --data_path /home/lxh/code/l2b_dividemix/cifar-10 --noise_mode 'sym' --r 0.5  --log_wandb=True --wandb_project='L2B_extend' --wandb_experiment='l2b_unicon_yes_clean_1000_cifar10_repro_nr50_lr002_batch64_seed0'