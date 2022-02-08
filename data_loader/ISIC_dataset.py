from PIL import Image
import os
import os.path
import numpy as np
import sys
import csv

import torch.utils.data as data

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributed as dist
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C


class ISIC2019(data.Dataset):

    def __init__(self, root='', train=True, meta=True, split='ISIC/train.lst', num_classes=8,
                 groundtruth_file='ISIC/ISIC_2019_Training_GroundTruth.csv',
                 corruption_prob=0, corruption_type='unif', transform=None, seed=1,  
                 local_rank=0, temporal_label_file='label_isic.txt'):
        self.root = root
        self.transform = transform
        self.train = train  
        self.meta = meta
        self.data_list = open(split).read().splitlines()
        self.corruption_prob = corruption_prob
        self.num = len(self.data_list) 
        self.num_classes = num_classes
        self.path = temporal_label_file


        # now load the picked numpy arrays
        if train:
            self.train_data = []
            self.train_labels = []
            self.noisy_or_not = []            

            with open(groundtruth_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if row[0] in self.data_list:
                        self.train_data += [os.path.join(self.root, row[0] + '.jpg')]
                        self.train_labels += [row.index("1.0") - 1]

            if not meta:
                if corruption_type == 'unif':
                    C = uniform_mix_C(self.corruption_prob, self.num_classes)
                   # if local_rank == 0:
                   #     print(C)
                    self.C = C
                elif corruption_type == 'flip':
                    C = flip_labels_C(self.corruption_prob, self.num_classes)
                   # if local_rank == 0:
                   #     print(C)
                    self.C = C
                elif corruption_type == 'flip2':
                    C = flip_labels_C_two(self.corruption_prob, self.num_classes)
                   # if local_rank == 0:
                    #    print(C)
                    self.C = C
                else:
                    assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(corruption_type)

                if is_main_process():
                    np.random.seed(seed)
                    for i in range(len(self.train_labels)):
                        original_label = self.train_labels[i]

                        self.train_labels[i] = np.random.choice(num_classes, p=C[self.train_labels[i]])

                        if original_label != self.train_labels[i]:
                            self.noisy_or_not.append(1.0)
                        else:
                            self.noisy_or_not.append(0.0)
                    file = open(self.path, 'w')
                    for l in self.train_labels:
                        file.write(str(l) + '\n')
                    file.close()
                else:
                    import time
                    time.sleep(20)
                    print('using the same noisy label')
                    if os.path.exists(self.path):
                        idx = 0
                        for i in open(self.path):
                            original_label = self.train_labels[idx]
                            self.train_labels[idx] = int(i)

                            if original_label != self.train_labels[idx]:
                                self.noisy_or_not.append(1.0)
                            else:
                                self.noisy_or_not.append(0.0)
                            idx += 1
                self.corruption_matrix = C
                print(str(100 * self.noisy_or_not.count(1.0) / len(self.noisy_or_not)) + '% data havs noise')

        else:
            self.test_data = []
            self.test_labels = []   
            with open(groundtruth_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    if row[0] in self.data_list:
                        self.test_data += [os.path.join(self.root, row[0] + '.jpg')]
                        self.test_labels += [row.index("1.0") - 1]

    def __getitem__(self, index):
        if not self.meta and self.train:
            image_path, target, noisy_or_not = self.train_data[index], self.train_labels[index], self.noisy_or_not[index]
        elif self.meta and self.train:
            image_path, target = self.train_data[index], self.train_labels[index]
        else:
            image_path, target = self.test_data[index], self.test_labels[index]

        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            
        if not self.meta and self.train:
            return img, target
        else:
            return img, target
            

    def __len__(self):
        return self.num
