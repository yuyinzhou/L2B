from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

class clothing_dataset(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, num_class=14, num_meta = 1400):
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}            
        
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/images/'%self.root+entry[0][7:]
                self.train_labels[img_path] = int(entry[1])                         
        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                img_path = '%s/images/'%self.root+entry[0][7:]
                self.test_labels[img_path] = int(entry[1])   

        if mode == 'all':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/images/'%self.root+l[7:]
                    train_imgs.append(img_path)                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)                                          
                         
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/images/'%self.root+l[7:]
                    self.test_imgs.append(img_path)            
        elif mode=='val':
            self.val_imgs = []
            class_num = torch.zeros(num_class)
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/images/'%self.root+l[7:]
                    label = self.test_labels[img_path]
                    #if class_num[label] < (num_meta / 14) and len(self.val_imgs) < num_meta:
                    self.val_imgs.append(img_path)
                    #    class_num[label] += 1
                    
    def __getitem__(self, index):  
         
        if self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target        
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            
        
