


# Example Run
After creating a virtual environment, run

	pip install -r requirements.txt

Example run (CIFAR10 with 50% symmetric noise) 

	python Train_cifar.py --dataset cifar10 --num_class 10 --data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 

Example run (CIFAR100 with 90% symmetric noise) 

	python Train_cifar.py --dataset cifar100 --num_class 100 --data_path ./data/cifar100 --noise_mode 'sym' --r 0.9 

This will throw an error as downloaded files will not be in proper folder. That is why they are needed to be manually moved to the "data_path".

Example Run (TinyImageNet with 50% symmetric noise)

	python Train_TinyImageNet.py --ratio 0.5


Example run (Clothing1M)
	
	python Train_clothing1M.py --batch_size 32 --num_epochs 200   

Example run (Webvision)
	
	python Train_webvision.py 



# L2B + UNICON
To reproduce our results with UNICON method:
first follow above instruction to prepare env and datasets,
please find  scripts with hyper-parameters located in <code> L2B_UNICON/exp/exp </code>

# Dataset
For datasets other than CIFAR10 and CIFAR100, you need to download them from their corresponsing website.

