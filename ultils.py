from collections import defaultdict, deque
import datetime
import time
import torch
import torch.nn.functional as F  
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

from data_loader import *


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()



def is_main_process():
    return get_rank() == 0


def soft_cross_entropy(logit, pseudo_target, reduction='none'):
    loss = -(pseudo_target * F.log_softmax(logit, -1)).sum(-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Invalid reduction: %s' % reduction)


def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
        self.avg = self.sum / (self.count + 1e-5)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def prepare_set(args):
    if args.dataset == 'clothing1m':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        trainset =  clothing_dataset(args.data_path,transform=transform_train, mode='all',num_samples=args.num_batch*args.train_batch_size)
        metaset = clothing_dataset(args.data_path,transform=transform_test, mode='val')
        valset = clothing_dataset(args.data_path,transform=transform_test, mode='val')
        testset =  clothing_dataset(args.data_path,transform=transform_test, mode='test')
        args.num_classes = 14
        return  trainset, metaset, valset, testset, args.num_classes

    if args.dataset == 'ISIC':
        transform_train = transforms.Compose([

            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),

            transforms.RandomResizedCrop(size=224, scale=(0.3, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),

            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if args.dataset == "cifar10":
        trainset = CIFAR10(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=transform_train,
            download=True, seed=args.seed, local_rank=args.local_rank,
            temporal_label_file=args.temp_label)

        metaset = CIFAR10(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=transform_train, download=True, seed=args.seed)

        testset = CIFAR10(root=args.data_path, train=False, transform=transform_test, download=True, seed=args.seed)

    if args.dataset == "cifar100":
        trainset = CIFAR100(
            root=args.data_path, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=transform_train, download=True, seed=args.seed,
            local_rank=args.local_rank,
            temporal_label_file=args.temp_label
        )

        metaset = CIFAR100(
            root=args.data_path, train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=transform_train, download=True, seed=args.seed)

        testset = CIFAR100(root=args.data_path, train=False, transform=transform_test, download=True, seed=args.seed)

    if args.dataset == 'ISIC':
        trainset = ISIC2019(
            root=args.data_path, train=True, meta=False, split='ISIC/train.lst',
            num_classes=args.num_classes, groundtruth_file='ISIC/ISIC_2019_Training_GroundTruth.csv',
            corruption_prob=args.corruption_prob, corruption_type=args.corruption_type, transform=transform_train, seed=args.seed,
            local_rank=args.local_rank,
            temporal_label_file=args.temp_label
        )
        metaset = ISIC2019(
            root=args.data_path, train=True, meta=True, split='ISIC/val.lst',
            num_classes=args.num_classes, groundtruth_file='ISIC/ISIC_2019_Training_GroundTruth.csv',
            corruption_prob=args.corruption_prob, corruption_type=args.corruption_type, transform=transform_train, seed=args.seed)

        testset = ISIC2019(root=args.data_path, train=False, split='ISIC/test.lst',
                           transform=transform_test, seed=args.seed)
    return trainset, metaset, testset

def prepare_dataloder(args):
    if args.dataset == 'clothing1m':
        trainset, metaset, valset, testset, args.num_classes = prepare_set(args)
    else:
        trainset, metaset, testset = prepare_set(args)

    if args.distribute:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True,
                                                                        num_replicas=torch.distributed.get_world_size(),
                                                                        rank=args.local_rank)
        if not args.baseline:
            meta_sampler = torch.utils.data.distributed.DistributedSampler(metaset, shuffle=True,
                                                                           num_replicas=torch.distributed.get_world_size(),
                                                                           rank=args.local_rank)
        else:
            meta_sampler = None
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=True,
                                                                       num_replicas=torch.distributed.get_world_size(),
                                                                       rank=args.local_rank)
        eval_batch_size = args.eval_batch_size // torch.distributed.get_world_size()
        train_batch_size = args.train_batch_size // torch.distributed.get_world_size()

    else:
        train_sampler = None
        meta_sampler = None
        test_sampler = None
        eval_batch_size = args.eval_batch_size
        train_batch_size = args.train_batch_size

    trainloader = torch.utils.data.DataLoader(trainset,
                                              shuffle=not args.distribute,
                                              batch_size=train_batch_size,
                                              num_workers=4,
                                              sampler=train_sampler,
                                              pin_memory=True)
    if not args.baseline:
        metaloader = torch.utils.data.DataLoader(metaset,
                                                 shuffle=not args.distribute,
                                                 batch_size=train_batch_size,
                                                 num_workers=0,
                                                 sampler=meta_sampler,
                                                 pin_memory=True)
    else:
        metaloader = None

    testloader = torch.utils.data.DataLoader(testset,
                                             shuffle=False,
                                             batch_size=eval_batch_size,
                                             num_workers=4,
                                             sampler=test_sampler,
                                             pin_memory=True)

    return trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler


