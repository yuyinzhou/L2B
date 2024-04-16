
## Results
Following tables summarize main results of the paper:

CIFAR-10:
![CIFAR-10 results](./img/cifar10.png)

CIFAR-100:
![CIFAR-100 results](./img/cifar100.png)

Clothing1M:
![Clothing1M results](./img/clothing.png)

mini-WebVision:
![mini-WebVision](./img/webvision.png)
## Running the code

First you need to install dependencies by running `pip install -r requirements.txt`.

You can download pretrained self-supervised models from 
[Google Drive](https://drive.google.com/drive/folders/1qYVdggtNFQZBZ-OqVJm80LBKUKpdLPUm?usp=sharing). 
Alternatively, you can train them by yourself, using [SimCLR implementation](https://github.com/HobbitLong/SupContrast).
Put them into `./pretrained` folder.

Then you can run the code for CIFAR
```
python3 main_cifar.py --r 0.8 --lambda_u 500 --dataset cifar100 --p_threshold 0.03 --data_path ./cifar-100 --experiment-name simclr_resnet18 --method selfsup --net resnet50
```
for Clothing1M
```
python3 main_clothing1M.py --data_path /path/to/clothing1m --experiment-name selfsup --method selfsup --p_threshold 0.7 --warmup 5 --num_epochs 120
```
or for mini-WebVision
```
python3 Train_webvision.py --p_threshold 0.03 --num_class 50 --data_path /path/to/webvision --imagenet_data_path /path/to/imagenet --method selfsup```
```

# L2B + C2D

To reproduce our results with C2D method:
first follow above instruction to prepare env and datasets,
please find  scripts with hyper-parameters located in <code> L2B_C2D/exp </code>
