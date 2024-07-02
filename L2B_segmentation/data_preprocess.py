import os
import random
import numpy as np
from scipy.io import loadmat
import cv2
from PIL import Image
from Metrics import dice_coeff


info = loadmat(r'./data_split/PROMISE12_data_split.mat')
loadp = r'./data/Prostate_data/all_data_prepro/'
corrupt_path = r'./data/Prostate_data/Corrupt_data_scan3_06/'
savep = r'./data/Prostate_data/l2b_train_scan3_corrupt_06/'
savep_meta = r'./data/Prostate_data/meta_train_scan3/'
os.makedirs(corrupt_path, exist_ok=True)
os.makedirs(savep, exist_ok=True)
os.makedirs(savep_meta, exist_ok=True)

meta_l2b_list = info['meta_l2b']
for k in meta_l2b_list:
    inf = np.load(loadp + k + '.npy', allow_pickle=True).item()
    imgall = inf['img']
    gtall = inf['label']
    for i in range(gtall.shape[0]):
        img = imgall[i, :, :]
        gt = gtall[i, :, :]
        sp = savep + k + '_' + str(i) + '.npy'
        np.save(sp, {'img': img, 'label': gt})

vali_l2b_list = info['train_l2b']
Sall = []
kernel = np.ones((7, 7), np.uint8)
kernel2 = np.ones((14, 14), np.uint8) # 06-level
kernel3 = np.ones((21, 21), np.uint8) # 06-level
kernel4 = np.ones((28, 28), np.uint8) # 06-level
kernel5 = np.ones((35, 35), np.uint8) # 06-level
# kernel4 = np.ones((49, 49), np.uint8) # 04-level
# kernel5 = np.ones((63, 63), np.uint8) # 04-level
# kernel2 = np.ones((11, 11), np.uint8) # 08-level
# kernel3 = np.ones((13, 13), np.uint8) # 08-level
# kernel4 = np.ones((17, 17), np.uint8) # 08-level
# kernel5 = np.ones((21, 21), np.uint8) # 08-level

ero_ratio = [0.1, 0.2, 0.5, 0.7, 1] #06-level
dil_ratio = [0.1, 0.2, 0.5, 0.7, 1] #06-level
#ero_ratio = [0.05, 0.1, 0.2, 0.6, 1] #04-level
#dil_ratio = [0.05, 0.1, 0.2, 0.6, 1] #04-level
# ero_ratio = [0.4, 0.6, 0.8, 0.9, 1] #08-level
# dil_ratio = [0.4, 0.6, 0.8, 0.9, 1] #08-level
rotate_range = [-10, 10]
Dice = []

for k in vali_l2b_list:
    inf = np.load(loadp + k + '.npy', allow_pickle=True).item()
    imgall = inf['img']
    gtall = inf['label']
    noisy_label = np.zeros_like(gtall)
    for i in range(gtall.shape[0]):
        gtori = gtall[i, :, :]
        img = imgall[i, :, :]
        if gtori.sum() > 300:
            gt = Image.fromarray(gtori * 255)
            gt = np.array(gt.rotate(random.randint(-20, 20))).astype(np.uint8)
            r_c = random.uniform(0, 1)
            r_p = random.uniform(0, 1)
            r_ec = random.uniform(0, 1)
            if r_c < 0.5:
                if r_p < ero_ratio[0]:
                    erosion = cv2.erode(gt, kernel)
                elif r_p < ero_ratio[1]:
                    erosion = cv2.erode(gt, kernel2)
                elif r_p < ero_ratio[2]:
                    erosion = cv2.erode(gt, kernel3)
                elif r_p < ero_ratio[3]:
                    erosion = cv2.erode(gt, kernel4)
                else:
                    erosion = cv2.erode(gt, kernel5)
                noisy_gt = erosion
            else:
                if r_p < dil_ratio[0]:
                    dilation = cv2.dilate(gt, kernel)
                elif r_p < dil_ratio[1]:
                    dilation = cv2.dilate(gt, kernel2)
                elif r_p < dil_ratio[2]:
                    dilation = cv2.dilate(gt, kernel3)
                elif r_p < dil_ratio[3]:
                    dilation = cv2.dilate(gt, kernel4)
                else:
                    dilation = cv2.dilate(gt, kernel5)
                noisy_gt = dilation
            noisy_gt = np.where(noisy_gt > 0, 1, 0)

            if r_ec < 0.5:
                if noisy_gt.sum() > 300:
                    newc = np.zeros_like(noisy_gt)
                    x = np.where(noisy_gt.sum(0) > 0)[0]
                    y = np.where(noisy_gt.sum(1) > 0)[0]
                    xc, yc = int((x[0] + x[-1]) / 2), int((y[0] + y[-1]) / 2)
                    rx, ry = int((x[-1] - x[0]) / 2), int((y[-1] - y[0]) / 2)
                    for m in range(gt.shape[0]):
                        for n in range(gt.shape[1]):
                            if ((m - yc) / ry) * ((m - yc) / ry) + ((n - xc) / rx) * ((n - xc) / rx) <= 1:
                                newc[m, n] = 1
                    noisy_gt = newc

            noisy_label[i, :, :] = noisy_gt
            a = 1
    dice = dice_coeff(noisy_label, gtall)
    print(dice)
    Dice.append(dice)
    np.save(corrupt_path + k + '.npy', noisy_label)

m = np.array(Dice).mean()
print("Corrupted dice:", m)

for k in vali_l2b_list:
    inf = np.load(loadp + k + '.npy', allow_pickle=True).item()
    imgall = inf['img']
    gtall = inf['label']
    noisy_label = np.load(corrupt_path + k + '.npy')
    for i in range(gtall.shape[0]):
        img = imgall[i, :, :]
        gt = gtall[i, :, :]
        nlabel = noisy_label[i, :, :]
        sp = savep + k + '_' + str(i) + '.npy'
        np.save(sp, {'img': img, 'label': gt, 'noisy_label': nlabel})



