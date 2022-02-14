# Learning to Bootstrap for Combating Label Noise

This repo is the official implementation of our paper ["Learning to Bootstrap for Combating Label Noise"](https://arxiv.org/pdf/2202.04291.pdf).

## Citation
If you use this code for your research, please cite our paper ["Learning to Bootstrap for Combating Label Noise"](https://arxiv.org/pdf/2202.04291.pdf).

```
@article{zhou2022learning,
  title   = {Learning to Bootstrap for Combating Label Noise}, 
  author  = {Yuyin Zhou and Xianhang Li and Fengze Liu and Xuxi Chen and Lequan Yu and Cihang Xie and Matthew P. Lungren and Lei Xing},
  journal = {arXiv preprint arXiv:2202.04291},
  year    = {2022},
}
```

## Requirements
Python >= 3.6.4 \
Pytorch >= 1.6.0 \
Higher = 0.2.1 \
Tensorboardx = 2.4.1


## Training
First, please create a folder to store checkpoints by using the following command.
```
mkdir checkpoint
```

#### CIFAR-10

To reproduce the results on CIFAR dataset from our paper, please follow the command and our hyper-parameters.

First, you can adjust the ``corruption_prob `` and ``corruption_type`` to obtain different noise rates and noise type.

Second, the ``reweight_label`` indicates you are using the our L2B method. You can change it to ``baseline`` or ``mixup``.

```python
python  main.py  --arch res18 --dataset cifar10 --num_classes 10 --exp L2B --train_batch_size  512 \
 --corruption_prob 0.2 --reweight_label  --lr 0.15  -clipping_norm 0.25  --num_epochs 300  --scheduler cos \
 --corruption_type unif  --warm_up 10  --seed 0  
```


#### CIFAR-100

Most of settings are the same as CIFAR-10. To reproduce the results, please follow the command.

```python
python  main.py  --arch res18 --dataset cifar100 --num_classes 100 --exp L2B --train_batch_size  256  \
--corruption_prob 0.2 --reweight_label  --lr 0.15  --clipping_norm 0.80  --num_epochs 300  --scheduler cos \
--corruption_type unif  --warm_up 10  --seed 0 \ 
```


#### ISIC2019

On the ISIC dataset, first you should download the dataset by following command.

Download ISIC dataset as follows:\
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip \
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv \


Then you can reproduce the results by following the command.

```python
python main.py  --arch res50  --dataset ISIC --data_path isic_data/ISIC_2019_Training_Input --num_classes 8 
--exp L2B  --train_batch_size 64  --corruption_prob 0.2 --lr 0.01 --clipping_norm 0.80 --num_epochs 30 
--temperature 10.0  --wd 5e-4  --scheduler cos --reweight_label --norm_type softmax --warm_up 1 
```



#### Clothing-1M

First, the ``num_batch`` and ``train_batch_size`` indicates how many training images you want to use (we sample a balanced training data for each epoch).

Second, you can adjust the ``num_meta`` to sample different numbers of validation images to form the metaset.  We use the whole validation set as metaset by default.

The ``data_path`` is where you store the data and key-label lists. And also change the data_path in the line 20 of ``main.py``.  If you have issue for downloading the dataset, please feel free to contact us.

Then you can reproduce the results by following the command.
```python
python main.py --arch res18_224 --num_batch 250 --dataset clothing1m \
--exp L2B_clothing1m_one_stage_multi_runs  --train_batch_size 256  --lr 0.005  \
--num_epochs 300  --reweight_label  --wd 5e-4 --scheduler cos   --warm_up 0 \
--data_path /data1/data/clothing1m/clothing1M  --norm_type org  --num_classes 14 \ 
--multi_runs 3 --num_meta 14313
```



## Contact

Yuyin Zhou
- email: yzhou284@ucsc.edu


Xianhang Li
- email: xli421@ucsc.edu


If you have any question about the code and data, please contact us directly.


