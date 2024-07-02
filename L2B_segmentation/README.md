# Apply L2B for segmentation under noisy supervision
## Data preparation
### PROMISE12 dataset
The original dataset can be downloaded from https://promise12.grand-challenge.org. You can also download our preprocessed data from https://drive.google.com/file/d/1Xaj9MuYxaqzA28rq8E3VZe6A9rZI5SvU/view?usp=share_link. And put it under `./data/Prostate_data/` folder.

### Synthetic noisy label creation
To generate synthetic noisy labels with corrupted ratios around 40%, please run the following script:
```
python data_proprocess.py
```

## Training under noisy supervision
To train a UNet++ model using PROMISE12 dataset with corrupted ratios around 40%, please run the following script:
```
python train.py --dataset Prostate --train_root ./data/Prostate_data/l2b_train_scan3_corrupt_06/  --meta_root ./data/Prostate_data/meta_train_scan3/ --vali_root ./data/Prostate_data/all_data_prepro/  --datasplitpath ./data_split/PROMISE12_data_split.mat
```
