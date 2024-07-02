import torch.utils.data as data
import numpy as np
import random
import torch
import torchvision
import os
import cv2
from scipy.io import loadmat
from PIL import Image


class Prostate_2D(data.Dataset):
    def __init__(self, root='', suffix='train', resize=[224, 224], transform=None, l2b=False, datasplitpath=''):
        self.root = root
        self.suffix = suffix
        self.transform = transform
        self.resize = resize

        self.l2b = l2b

        self.train_pick_frame = None
        self.data_split_info = loadmat(datasplitpath)

        if self.l2b:
            if suffix == 'train':
                self.images_names = os.listdir(self.root)
            else:
                if suffix == 'vali':
                    self.images_names = self.data_split_info['test']
                else:
                    self.images_names = os.listdir(self.root)
        else:
            if suffix == 'train':
                self.images_names = os.listdir(self.root)
            else:
                self.images_names = self.data_split_info['test']
        if self.transform is None:
            if suffix == 'train' or suffix == 'meta':
                self.tx = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(self.resize),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation((-10, 10)),
                    torchvision.transforms.ToTensor()
                ])
                self.lx = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(resize),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation((-10, 10)),
                    torchvision.transforms.ToTensor(),
                ])
            else:
                self.tx = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(resize),
                    torchvision.transforms.ToTensor(),
                ])
                self.lx = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(resize),
                    torchvision.transforms.ToTensor()
                ])
        self.num = len(self.images_names)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        image_name = self.images_names[index]

        if 'meta' in self.suffix:
            image_path = self.root + image_name
            data = np.load(image_path, allow_pickle=True).item()
            images = np.expand_dims(data['img'], 0)
            masksgt = np.expand_dims(data['label'], 0)
            masks = masksgt
        else:
            if 'train' in self.suffix:
                image_path = self.root + image_name
                data = np.load(image_path, allow_pickle=True).item()
                images = np.expand_dims(data['img'], 0)
                masksgt = np.expand_dims(data['label'], 0)
                masks = np.expand_dims(data['noisy_label'], 0)
            else:
                if '.npy' not in image_name:
                    image_path = self.root + image_name + '.npy'
                else:
                    image_path = self.root + image_name
                data = np.load(image_path, allow_pickle=True).item()

                images = data['img']
                masksgt = data['label']
                masks = masksgt

        masks = np.where(masks > 0, 1, 0)
        masksgt = np.where(masksgt > 0, 1, 0)


        if self.transform is not None:
            images, masks = self.transform.transform(images, masks)
        else:
            if self.suffix == 'vali':
                new_im, new_im1, new_im2 = [], [], []
                new_la, new_la1, new_la2 = [], [], []
                orimask = masks
                for img, label in zip(images, masks):
                    seed = np.random.randint(0, 2 ** 32, dtype='int64')

                    rmin = img.min()
                    rmax = img.max()
                    ra = rmax - rmin
                    img = (((img - rmin) / ra) * 255).astype(np.uint8)
                    img = Image.fromarray(np.expand_dims(img, 2).repeat(3, 2))

                    label = label * 255
                    label = np.expand_dims(label, 2).repeat(3, 2)
                    label = Image.fromarray(label.astype(np.uint8))

                    random.seed(seed)
                    torch.manual_seed(seed)
                    img = self.tx(img)[0, :, :]

                    random.seed(seed)
                    torch.manual_seed(seed)
                    label = self.lx(label)[0, :, :]
                    new_im.append(img)
                    new_la.append(label > 0)
                images = torch.stack(new_im)
                masks = torch.stack(new_la)
            else:
                if self.l2b & (self.suffix != 'meta'):
                    new_im = []
                    new_la = []
                    new_gt = []
                    for img, label, gt in zip(images, masks, masksgt):
                        seed = np.random.randint(0, 2 ** 32, dtype='int64')

                        rmin = img.min()
                        rmax = img.max()
                        ra = rmax - rmin
                        img = (((img - rmin) / ra) * 255).astype(np.uint8)
                        img = Image.fromarray(np.expand_dims(img, 2).repeat(3, 2))

                        label = label * 255
                        label = np.expand_dims(label, 2).repeat(3, 2)
                        label = Image.fromarray(label.astype(np.uint8))

                        gt = np.expand_dims(gt, 2).repeat(3, 2)
                        gt = Image.fromarray(gt.astype(np.uint8))

                        random.seed(seed)
                        torch.manual_seed(seed)
                        img = self.tx(img)[0, :, :]

                        random.seed(seed)
                        torch.manual_seed(seed)
                        label = self.lx(label)[0, :, :]
                        random.seed(seed)
                        torch.manual_seed(seed)
                        gt = self.lx(gt)[0, :, :]
                        new_im.append(img)
                        new_la.append(label > 0)
                        new_gt.append(gt > 0)

                    images = torch.stack(new_im)
                    masks = torch.stack(new_la)
                    gts = torch.stack(new_gt)
                else:
                    new_im = []
                    new_la = []
                    for img, label in zip(images, masks):
                        seed = np.random.randint(0, 2 ** 32, dtype='int64')

                        rmin = img.min()
                        rmax = img.max()
                        ra = rmax - rmin
                        img = (((img - rmin) / ra) * 255).astype(np.uint8)
                        label = label * 255
                        img = Image.fromarray(np.expand_dims(img, 2).repeat(3, 2))
                        label = np.expand_dims(label, 2).repeat(3, 2)
                        label = Image.fromarray(label.astype(np.uint8))

                        random.seed(seed)
                        torch.manual_seed(seed)
                        img = self.tx(img)[0, :, :]

                        random.seed(seed)
                        torch.manual_seed(seed)
                        label = self.lx(label)[0, :, :]

                        new_im.append(img)
                        new_la.append(label > 0)

                    images = torch.stack(new_im)
                    masks = torch.stack(new_la)

        if self.l2b:
            if self.suffix == 'train':
                return images, masks.float(), gts.float()
            else:
                if self.suffix == 'meta':
                    return images, masks.float()
                else:
                    return images, masks.float(), orimask, image_name
        else:
            if self.suffix == 'vali':
                return images, masks.float(), orimask
            else:
                return images, masks.float()



def prepare_set2D_l2b(args):
    if args.dataset == 'Prostate':
        trainset = Prostate_2D(root=args.train_root, suffix='train', resize=args.size, transform=None, l2b=True,
                         datasplitpath=args.datasplitpath)
        metaset = Prostate_2D(root=args.meta_root, suffix='meta', resize=args.size, transform=None, l2b=True,
                        datasplitpath=args.datasplitpath)
        testset = Prostate_2D(root=args.vali_root, suffix='vali', resize=args.size, transform=None, l2b=True,
                        datasplitpath=args.datasplitpath)

    else:
        trainset, metaset, testset = None, None, None

    return trainset, metaset, testset



def prepare_dataloder(args):
    trainset, metaset, testset = prepare_set2D_l2b(args)
    meta_sampler = None
    train_sampler = None
    test_sampler = None
    eval_batch_size = args.eval_batch_size
    train_batch_size = args.train_batch_size
    meta_batch_size = args.meta_batch_size

    trainloader = torch.utils.data.DataLoader(trainset,
                                              shuffle=not args.distribute,
                                              batch_size=train_batch_size,
                                              num_workers=4,
                                              sampler=train_sampler,
                                              pin_memory=True)


    testloader = torch.utils.data.DataLoader(testset,
                                             shuffle=False,
                                             batch_size=eval_batch_size,
                                             num_workers=4,
                                             sampler=test_sampler,
                                             pin_memory=True)


    metaloader = torch.utils.data.DataLoader(metaset,
                                             shuffle=not args.distribute,
                                             batch_size=meta_batch_size,
                                             num_workers=4,
                                             sampler=meta_sampler,
                                             pin_memory=True)


    return trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def get_aug_image(image, gt, gtmasks, picksize=[5, 10, 15, 20]):
    AUG = 5
    b, c, w, h = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
    image_aug = torch.zeros([b*AUG, c, w, h]).cuda()
    gt_aug = torch.zeros([b*AUG, 1, w, h]).cuda()
    gtmasks_aug = torch.zeros([b * AUG, 1, w, h]).cuda()
    insize = []
    outsize = []
    flipindex = []

    flipchoice = [1, 2, 3]

    for i in range(image.shape[0]):
        imgall = torch.zeros([AUG, c, w, h]).cuda()
        gtall = torch.zeros([AUG, 1, w, h]).cuda()
        gtmasksall = torch.zeros([AUG, 1, w, h]).cuda()
        imgall[0, :, :, :] = image[i, :, :, :]
        gtall[0, :, :, :] = gt[i, :, :, :]
        gtmasksall[0, :, :, :] = gtmasks[i, :, :, :]

        imgori = image[i, 0, :, :].cpu().detach().numpy()
        gtori = gt[i, 0, :, :].cpu().detach().numpy()
        gtmasksori = gtmasks[i, 0, :, :].cpu().detach().numpy()


        padsizein = random.sample(picksize, 1)[0]
        insize.append(padsizein)
        reshape = [h + 2 * padsizein, w + 2 * padsizein]
        img = cv2.resize(imgori, reshape)
        gtimg = cv2.resize(gtori, reshape)
        gtmasksimg = cv2.resize(gtmasksori, reshape)
        imgr = img[padsizein:-padsizein, padsizein:-padsizein]
        gtimgr = gtimg[padsizein:-padsizein, padsizein:-padsizein]
        gtmasksimgr = gtmasksimg[padsizein:-padsizein, padsizein:-padsizein]
        imgall[1, :, :, :] = torch.tensor(imgr).cuda()
        gtall[1, :, :, :] = torch.tensor(gtimgr).cuda()
        gtmasksall[1, :, :, :] = torch.tensor(gtmasksimgr).cuda()


        padsizeoutall = random.sample(picksize, 2)
        outsize.append(padsizeoutall)
        padsizeout = padsizeoutall[0]
        reshape = [h - 2 * padsizeout, w - 2 * padsizeout] 
        img = cv2.resize(imgori, reshape)
       
        imgr = np.zeros([w, h]) #+ img.min()
        gtimg = cv2.resize(gtori, reshape)
        gtimgr = np.zeros([w, h])
        gtmasksimg = cv2.resize(gtmasksori, reshape)
        gtmasksimgr = np.zeros([w, h])
        imgr[padsizeout:-padsizeout, padsizeout:-padsizeout] = img
        gtimgr[padsizeout:-padsizeout, padsizeout:-padsizeout] = gtimg
        gtmasksimgr[padsizeout:-padsizeout, padsizeout:-padsizeout] = gtmasksimg
        imgall[2, :, :, :] = torch.tensor(imgr).cuda()
        gtall[2, :, :, :] = torch.tensor(gtimgr).cuda()
        gtmasksall[2, :, :, :] = torch.tensor(gtmasksimgr).cuda()

        padsizeout = padsizeoutall[1]
        reshape = [h - 2 * padsizeout, w - 2 * padsizeout] 
        img = cv2.resize(imgori, reshape)
       
        imgr = np.zeros([w, h])  
        gtimg = cv2.resize(gtori, reshape)
        gtimgr = np.zeros([w, h])
        gtmasksimg = cv2.resize(gtmasksori, reshape)
        gtmasksimgr = np.zeros([w, h])
        imgr[padsizeout:-padsizeout, padsizeout:-padsizeout] = img
        gtimgr[padsizeout:-padsizeout, padsizeout:-padsizeout] = gtimg
        gtmasksimgr[padsizeout:-padsizeout, padsizeout:-padsizeout] = gtmasksimg
        imgall[3, :, :, :] = torch.tensor(imgr).cuda()
        gtall[3, :, :, :] = torch.tensor(gtimgr).cuda()
        gtmasksall[3, :, :, :] = torch.tensor(gtmasksimgr).cuda()

        findex = random.sample(flipchoice, 1)#[0]
        flipindex.append(findex)

        if findex[0] == 3:
            imgall[4, :, :, :] = flip(flip(image[i, :, :, :], 1), 2)
            gtall[4, :, :, :] = flip(flip(gt[i, :, :, :], 1), 2)
            gtmasksall[4, :, :, :] = flip(flip(gtmasks[i, :, :, :], 1), 2)
        else:
            imgall[4, :, :, :] = flip(image[i, :, :, :], findex[0])
            gtall[4, :, :, :] = flip(gt[i, :, :, :], findex[0])
            gtmasksall[4, :, :, :] = flip(gtmasks[i, :, :, :], findex[0])



        image_aug[i * AUG:(i + 1) * AUG, :, :, :] = imgall
        gt_aug[i * AUG:(i + 1) * AUG, :, :, :] = gtall
        gtmasks_aug[i * AUG:(i + 1) * AUG, :, :, :] = gtmasksall

    return image_aug, (gt_aug > 0.5) * 1.0, (gtmasks_aug > 0.5) * 1.0, insize, outsize, flipindex


def avg_pseudo_label(pseudo_label_all, insize, outsize, flipindex):
    AUG = 5
    b, c, w, h = pseudo_label_all.shape[0], pseudo_label_all.shape[1], pseudo_label_all.shape[2], pseudo_label_all.shape[3]
    pseudo_label_avg_all = torch.zeros([b, c, w, h]).cuda()
    for i in range(b // AUG):
        zoom_in, zoom_outall, findex = insize[i], outsize[i], flipindex[i]
        pseudo_label = pseudo_label_all[i*AUG:i*AUG+AUG, :, :, :]
        p_0 = pseudo_label[0, :, :, :]

        p_final = p_0[0, :, :]


        p_2 = pseudo_label[1, :, :, :]
        reshape = [w + 2 * zoom_in, h + 2 * zoom_in]
        p_2_ori = torch.zeros(reshape)
        p_2_ori[zoom_in:-zoom_in, zoom_in:-zoom_in] = p_2
        p_2 = cv2.resize(p_2_ori.numpy(), [h, w])
        p_final = p_final + torch.tensor(p_2).cuda()

        p_3 = pseudo_label[2, :, :, :]
        zoom_out = zoom_outall[0]
        p_3_ori = p_3[0, zoom_out:-zoom_out, zoom_out:-zoom_out]
        p_3 = cv2.resize(p_3_ori.cpu().numpy(), [h, w])
        p_final = p_final + torch.tensor(p_3).cuda()
        #print(findex)

        p_4 = pseudo_label[3, :, :, :]
        zoom_out = zoom_outall[1]
        p_4_ori = p_4[0, zoom_out:-zoom_out, zoom_out:-zoom_out]
        p_4 = cv2.resize(p_4_ori.cpu().numpy(), [h, w])
        p_final = p_final + torch.tensor(p_4).cuda()

        p_5 = pseudo_label[4, :, :, :]
        if findex[0] == 3:
            p_5 = flip(flip(p_5, 2), 1)
        else:
            p_5 = flip(p_5, findex[0])
        p_final = p_final + p_5[0, :, :]

        p_final = p_final / AUG
        pseudo_label_avg_b = torch.zeros_like(pseudo_label).cuda()
        pseudo_label_avg_b[0, :, :, :] = p_final


        reshape = [h + 2 * zoom_in, w + 2 * zoom_in]
        img = cv2.resize(p_final.cpu().numpy(), reshape)
        imgr = img[zoom_in:-zoom_in, zoom_in:-zoom_in]
        pseudo_label_avg_b[1, :, :, :] = torch.tensor(imgr).cuda()

        zoom_out = zoom_outall[0]
        reshape = [h - 2 * zoom_out, w - 2 * zoom_out]
        img = cv2.resize(p_final.cpu().numpy(), reshape)
        imgr = np.zeros([w, h])
        imgr[zoom_out:-zoom_out, zoom_out:-zoom_out] = img
        pseudo_label_avg_b[2, :, :, :] = torch.tensor(imgr).cuda()

        zoom_out = zoom_outall[1]
        reshape = [h - 2 * zoom_out, w - 2 * zoom_out]
        img = cv2.resize(p_final.cpu().numpy(), reshape)
        imgr = np.zeros([w, h])
        imgr[zoom_out:-zoom_out, zoom_out:-zoom_out] = img
        pseudo_label_avg_b[3, :, :, :] = torch.tensor(imgr).cuda()


        if findex[0] == 3:
            pseudo_label_avg_b[4, :, :, :] = flip(flip(p_final.unsqueeze(0), 1), 2)
        else:
            pseudo_label_avg_b[4, :, :, :] = flip(p_final.unsqueeze(0), findex[0])


        pseudo_label_avg_all[i*AUG:i*AUG+AUG, :, :, :] = pseudo_label_avg_b



    return pseudo_label_avg_all


