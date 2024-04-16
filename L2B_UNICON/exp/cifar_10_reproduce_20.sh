

source activate lxh_l2b
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install  scikit-learn higher torchnet

CUDA_VISIBLE_DEVICES=0 python Train_cifar.py --dataset cifar10 --num_class 10 --num_epochs=300 --data_path /home/lxh/code/l2b_dividemix/cifar-10 --noise_mode 'sym' --r 0.2 --lambda_u 0