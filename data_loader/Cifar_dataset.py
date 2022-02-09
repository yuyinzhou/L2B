import time

from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

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


class CIFAR10(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root='', train=True, meta=True, num_meta=1000,
                 corruption_prob=0, corruption_type='unif', transform=None, target_transform=None,
                 download=False, seed=1, local_rank=0, temporal_label_file='label_cifar.txt'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.meta = meta
        self.corruption_prob = corruption_prob
        self.num_meta = num_meta
        self.path = temporal_label_file

        if download and is_main_process():
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.noisy_or_not = []
            self.train_coarse_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                    img_num_list = [int(self.num_meta / 10)] * 10
                    self.num_classes = 10
                else:
                    self.train_labels += entry['fine_labels']
                    self.train_coarse_labels += entry['coarse_labels']
                    img_num_list = [int(self.num_meta / 100)] * 100
                    self.num_classes = 100
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            data_list_val = {}
            for j in range(self.num_classes):
                data_list_val[j] = [i for i, label in enumerate(self.train_labels) if label == j]

            idx_to_meta = []
            idx_to_train = []
            #if local_rank == 0:
            #    print(img_num_list)
            np.random.seed(seed)
            for cls_idx, img_id_list in data_list_val.items():
                np.random.shuffle(img_id_list)
                img_num = img_num_list[int(cls_idx)]
                idx_to_meta.extend(img_id_list[:img_num])
                idx_to_train.extend(img_id_list[img_num:])
            #if local_rank == 0:
             #   print(len(idx_to_meta), len(idx_to_train))
            self.train_idx = idx_to_train

            if meta is True:
                self.train_data = self.train_data[idx_to_meta]
                self.train_labels = list(np.array(self.train_labels)[idx_to_meta])
            else:
                self.train_data = self.train_data[idx_to_train]
                self.train_labels = list(np.array(self.train_labels)[idx_to_train])
                if corruption_type == 'hierarchical':
                    self.train_coarse_labels = list(np.array(self.train_coarse_labels)[idx_to_meta])

                if corruption_type == 'unif':
                    C = uniform_mix_C(self.corruption_prob, self.num_classes)
                    #if local_rank == 0:
                    #    print(C)
                    self.C = C
                elif corruption_type == 'flip':
                    C = flip_labels_C(self.corruption_prob, self.num_classes)
                    #if local_rank == 0:
                    #    print(C)
                    self.C = C
                elif corruption_type == 'flip2':
                    C = flip_labels_C_two(self.corruption_prob, self.num_classes)
                    #if local_rank == 0:
                    #    print(C)
                    self.C = C
                elif corruption_type == 'hierarchical':
                    assert num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
                    coarse_fine = []
                    for i in range(20):
                        coarse_fine.append(set())
                    for i in range(len(self.train_labels)):
                        coarse_fine[self.train_coarse_labels[i]].add(self.train_labels[i])
                    for i in range(20):
                        coarse_fine[i] = list(coarse_fine[i])

                    C = np.eye(num_classes) * (1 - corruption_prob)

                    for i in range(20):
                        tmp = np.copy(coarse_fine[i])
                        for j in range(len(tmp)):
                            tmp2 = np.delete(np.copy(tmp), j)
                            C[tmp[j], tmp2] += corruption_prob * 1 / len(tmp2)
                    self.C = C
                    #print(C)
                else:
                    assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(
                        corruption_type)

                if is_main_process() and not os.path.exists(self.path):
                    print(self.train_idx[:10])

                    num_noise = int(self.corruption_prob * len(idx_to_train))
                    noise_idx = idx_to_train[:num_noise]
                    #print(len(noise_idx))
                    for i in range(len(self.train_labels)):
                        original_label = self.train_labels[i]
                        if corruption_type == 'unif':
                            #if i in noise_idx:
                                  # if corruption_type == 'unif':
                                # self.train_labels[i] = np.random.randint(0, self.num_classes - 1)
                            self.train_labels[i] = np.random.choice(self.num_classes, p=C[self.train_labels[i]])

                        elif corruption_type == 'flip':
                            self.train_labels[i] = np.random.choice(self.num_classes, p=C[self.train_labels[i]])
                        if original_label != self.train_labels[i]:
                            self.noisy_or_not.append(1.0)
                        else:
                            self.noisy_or_not.append(0.0)
                    print(self.train_labels[:10])
                    file = open(self.path, 'w')
                    for l in self.train_labels:
                        file.write(str(l) + '\n')
                    file.close()
                    print(str(100 * self.noisy_or_not.count(1.0) / len(self.noisy_or_not)) + '% data havs noise')
                else:
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
                    print(self.train_idx[:10])
                    print(self.train_labels[:10])
                    print(str(100 * self.noisy_or_not.count(1.0) / len(self.noisy_or_not)) + '% data havs noise')

                self.corruption_matrix = C
        else:
            f = self.test_list[0][0]
            file = os.path.join(root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        if not self.meta and self.train:
            img, target, noisy_or_not = self.train_data[index], self.train_labels[index], self.noisy_or_not[index]
        elif self.meta and self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.meta and self.train:
            return img, target#, self.train_idx[index], self.noisy_or_not[index]
        else:
            return img, target

    def __len__(self):
        if self.train:
            if self.meta is True:
                return self.num_meta
            else:
                return 50000 - self.num_meta
        else:
            return 10000

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            return
            #print('Files already downloaded and verified')
            #return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
