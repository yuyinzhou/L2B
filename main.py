import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import higher
import argparse
import os
from models import *
import time
from tensorboardX import SummaryWriter
from ultils import *


model_names = ['res18', 'res50', 'wrn28_10', 'res18_224']
data_path = ['./data', 'isic_data/ISIC_2019_Training_Input', '/data1/data/clothing1m/clothing1M']

parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--multi_runs', default=1, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--reweight_label', action='store_true')
parser.add_argument('--exp', type=str,
                    default='baseline_cifar10', help='exp name')
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_meta', default=1000, type=int)
parser.add_argument('--gpuid', default="0", type=str)
parser.add_argument('--train_batch_size', default=256, type=int,
                    help="Total batch size for training.")
parser.add_argument('--eval_batch_size', default=128, type=int,
                    help="Total batch size for eval.")
parser.add_argument('--corruption_prob', default=0.2, type=float, help='corruption rate')
parser.add_argument('--corruption_type', default='unif', type=str, help='corruption type')
parser.add_argument('--norm_type', default='org', type=str, help='normalization type')
parser.add_argument('--clipping_norm', default=0, type=float, help='')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_batch', type=int, default=250)
parser.add_argument('--ngpu', type=int, default=3, help='0 = CPU')
parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument('--distribute', action='store_true')
parser.add_argument('--arch', metavar='ARCH', default='wrn28_10', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: wrn28_10)')
parser.add_argument('--data_path', metavar='ARCH', default='./data', choices=data_path,
                    help='model architecture: ' + ' | '.join(data_path) + ' (default: ./data)')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay for SGD')
parser.add_argument('--temp_label', type=str,
                    help='temporal_label txt name')
parser.add_argument('--meta_opt', type=str, default='sgd',
                    help='type of meta optimizer')
parser.add_argument('--meta_lr', type=float, default=1e-2,
                    help='learning rate of meta optimizer')
parser.add_argument('--meta_step', type=int, default=1,
                    help='training step of meta optimizer')
parser.add_argument('--print_freq', type=int, default=50,
                    help='print frequency')
parser.add_argument('--scheduler', type=str, default='step',
                    help='type of scheduler')
parser.add_argument('--temperature', default=1.0, type=float, help='temperature for softmax norm')

parser.add_argument('--milestone', type=str, nargs='+', default=[150],
                    help='milestone of step scheduler')

parser.add_argument('--warm_up', type=int, default=1,
                    help='warm up training with CE')
parser.add_argument('--hard_weight', action='store_true')
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint')
parser.add_argument('--no_val_data', action='store_true',
                    help='using validation set or not')

def main(args):  
      global best_acc
      best_acc = 0
      start_time = time.time()
      current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
      exp = args.arch + '_' +'temperature_'+str(args.temperature)+\
                 '_'+'corruption_'+str(args.corruption_prob)+ '_'+\
                 'epochs_'+str(args.num_epochs)+ '_'+\
                 'dataset_'+str(args.dataset)+ '_'+args.exp
      if not os.path.exists(os.path.join('checkpoint', exp)) and is_main_process():
           os.mkdir(os.path.join('checkpoint', exp))
      exp = os.path.join(exp, current_time.strip().replace(' ', '-'))
      writer = None
      if not os.path.exists(os.path.join('checkpoint', exp)) and is_main_process():
          os.mkdir(os.path.join('checkpoint', exp))
          writer = SummaryWriter(log_dir=os.path.join('logs/', exp))       
          print('Experiments is conducted on:', exp)
          print('==> Preparing data..')
      if args.local_rank == 0:
          print(args)
          if args.mixup:
              print('Using Mixup For training')

      if args.arch == 'res50':
          net = ResNet50(num_classes=args.num_classes).cuda(device=args.gpuid)

      elif args.arch == 'res18':
          net = PreActResNet18(num_classes=args.num_classes).cuda(device=args.gpuid)
      elif args.arch == 'wrn28_10':
          net = wrn28_10(num_classes=args.num_classes).cuda(device=args.gpuid)
      elif args.arch == 'res18_224':
          net = ResNet18(num_classes=args.num_classes)
      if args.resume:
          path = args.resume
          dict = torch.load(path, map_location='cpu')['net']
          net_dict = net.state_dict()
          idy = 0
          for k, v in dict.items():
              k = k.replace('module.', '')
              if k in net_dict:
                  net_dict[k] = v
                  idy += 1
          print(len(net_dict), idy, 'update state dict already')
          net.load_state_dict(net_dict)
      net = net.cuda(device=args.gpuid)

      if args.distribute:
          net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpuid])

      if args.baseline:
          print('using the norm CE loss')
          criterion = nn.CrossEntropyLoss().cuda()
      else:
          criterion = nn.CrossEntropyLoss(reduction="none").cuda(args.gpuid)

      optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

      if args.meta_opt == 'adam':
          optimizer_meta = optim.Adam(net.parameters(), lr=args.meta_lr, weight_decay=0,
                                      # args.wdecay, # meta should have wdecay or not??
                                      amsgrad=True, eps=1e-8)

      elif args.meta_opt == 'sgd':
          optimizer_meta = optim.SGD(net.parameters(), lr=args.meta_lr, momentum=0.9, weight_decay=0)
      if args.scheduler =='cos':
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
      elif  args.scheduler =='step':
          args.milestone =[int(step) for step in args.milestone]
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.1)

      args.temp_label = args.dataset + 'label.txt'
      if os.path.exists(args.temp_label) and is_main_process():
          os.remove(args.temp_label)
          print('remove the temp label file already')
      if not args.dataset =='clothing1m':
          trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler = prepare_dataloder(args)

      if args.eval:
          trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler = prepare_dataloder(args)
          acc, test_loss = val(0, testloader, net, writer, args)
          print('-----------------------------------------------------')
          print(' CURRENT loss: {:.4f}'.format(test_loss.avg))
          print(' BEST accuracy: {:.4f}'.format(acc))
          print('-----------------------------------------------------')
          return

      for epoch in range(args.num_epochs):
          if  args.dataset == 'clothing1m':
              if not args.no_val_data:
                  trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler = prepare_dataloder(args)

              else:
                  if is_main_process():
                      print('using pseudo labeled training subset for training')
                  trainloader, testloader, metaloader, train_sampler, test_sampler, meta_sampler = prepare_dataloder_clothing1m(args)
          train(epoch, trainloader, metaloader, train_sampler, net, meta_sampler, test_sampler,
                optimizer, optimizer_meta, criterion,  writer , args)
          acc, test_loss = val(epoch, testloader,  net, writer, args )

          scheduler.step()
          if acc > best_acc :
              if is_main_process():
                 print('Saving..')
                 state = {
                     'net': net.state_dict(),
                     'acc': acc,
                     'epoch': epoch,
                 }
                 if not os.path.isdir('checkpoint'):
                     os.mkdir('checkpoint')
                 torch.save(state, './checkpoint/{}_ckpt.pth'.format(exp))
                 best_acc = acc

          if is_main_process():
              print('-----------------------------------------------------')
              print('At epoch: {:03d} CURRENT loss: {:.4f}'.format(epoch, test_loss.avg))
              print('At epoch: {:03d} BEST accuracy: {:.4f}'.format(epoch, best_acc))
              print('At epoch: {:03d} CURRENT accuracy: {:.4f}'.format(epoch, acc))
              print('-----------------------------------------------------')
      if is_main_process():
          current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
          end_time = time.time()
          print(current_time, end_time-start_time)
      return  best_acc





def train(epoch, trainloader, metaloader, train_sampler, net, meta_sampler, test_sampler,
          optimizer, optimizer_meta, criterion,  writer , args):
    
    if args.distribute:
        train_sampler.set_epoch(epoch)
        if not args.baseline:
            meta_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
    net.train()
    train_loss = AverageMeter()
    train_top1 = AverageMeter()

    for i, (image, labels) in enumerate(trainloader):
        image = image.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        org_label = labels   
        
        if args.mixup:          
            lam = np.random.beta(32.0, 32.0)
            indices = np.random.permutation(image.size(0))
            image = image * lam + image[indices] * (1 - lam)        
            labels_shuffel = labels[indices]        
            
        if args.baseline :
            outputs = net(image)
            if args.mixup:
                l_f = criterion(outputs, labels)*lam + criterion(outputs, labels_shuffel)*(1 - lam) 
            else:
                l_f =criterion(outputs, labels)
            prec1, prec5 = accuracy(outputs, org_label, topk=(1, 5))
        else:
            
            if epoch <= (args.warm_up-1):
                outputs = net(image)

                criterion_warm = nn.CrossEntropyLoss().cuda()
                if args.mixup:                  
                    l_f = criterion_warm(outputs, labels) * lam + criterion_warm(outputs, labels_shuffel) * (1 - lam)
                else:
                    l_f = criterion_warm(outputs, labels)

                prec1, prec5 = accuracy(outputs, org_label, topk=(1, 5))

            else:
                with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_optimizer):
                    for s in range(args.meta_step):
                        y_f_hat = meta_net(image)
                        _, pesudo_labels = y_f_hat.max(1)
                        if args.reweight_label:
                            if args.mixup:
                                cost1 = criterion(y_f_hat, labels) * lam + criterion(y_f_hat, labels_shuffel) * (1 - lam)
                                cost2 = criterion(y_f_hat, pesudo_labels)

                            else:
                                cost1 = criterion(y_f_hat, labels)
                                cost2 = criterion(y_f_hat, pesudo_labels)
                            cost = torch.cat((cost1.unsqueeze(0), cost2.unsqueeze(0)), dim=0)
                        else:

                            if args.mixup:
                                cost =  criterion(y_f_hat, labels) * lam + criterion(y_f_hat, labels_shuffel) * (1 - lam)
                            else:
                                cost = criterion(y_f_hat, labels)

                        eps = torch.zeros(cost.size()).cuda()
                        eps = eps.requires_grad_()

                        l_f_meta = torch.sum(cost * eps)

                        meta_optimizer.step(l_f_meta)


                    val_data, val_labels = next(iter(metaloader))
                    val_data = val_data.cuda()
                    val_labels = val_labels.cuda()

                    y_g_hat = meta_net(val_data)
                    l_g_meta = torch.mean(criterion(y_g_hat, val_labels))

                    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()

                if not args.reweight_label:
                    w_tilde = torch.clamp(-grad_eps, min=0)
                    if args.dataset =='ISIC':
                        w_tilde  = torch.sigmoid(w_tilde)
                    norm_c = torch.sum(w_tilde) + 1e-10

                    if norm_c != 0:
                        weight = w_tilde / norm_c
                    else:
                        weight = w_tilde
                else:
                    w_tilde = -grad_eps
                    w_tilde = w_tilde.view(-1)
                    
                    if args.hard_weight:
                        w_tilde = F.softmax(w_tilde , dim=-1)
                        zeros = torch.zeros_like(w_tilde).cuda()
                        weight = torch.where(w_tilde<= args.temperature, zeros, w_tilde )
                    else:

                       # $weight = F.softmax(w_tilde/args.temperature, dim=-1)
                       if args.norm_type=='org':
                            w_tilde = torch.clamp(w_tilde, min=0)
                            #if args.dataset == 'ISIC':
                            #    w_tilde = torch.sigmoid(w_tilde)
                            norm_c = torch.sum(w_tilde) + 1e-10

                            if norm_c != 0:
                                weight = w_tilde / norm_c
                            else:
                                weight = w_tilde
                       elif  args.norm_type=='softmax':
                             weight = F.softmax(w_tilde/args.temperature, dim=-1)
                       elif  args.norm_type== 'sigmoid':
                             sigmoid = nn.Sigmoid()
                             weight = sigmoid(w_tilde)
                             weight /= torch.sum(weight)
                       

                    weight = weight.view(2, -1)

                y_f_hat = net(image)
                _, pesudo_labels = y_f_hat.max(1)
                prec1, prec5 = accuracy(y_f_hat, org_label, topk=(1, 5))
                
                if args.reweight_label:
                    if args.mixup:
                        cost1 = criterion(y_f_hat, labels) * lam + criterion(y_f_hat, labels_shuffel) * (1 - lam)
                        cost2 = criterion(y_f_hat, pesudo_labels)
                    else:
                        cost1 = criterion(y_f_hat, labels)
                        cost2 = criterion(y_f_hat, pesudo_labels)
                        
                    cost = torch.cat((cost1.unsqueeze(0), cost2.unsqueeze(0)), dim=0)

                else:
                    if args.mixup:
                        cost = criterion(y_f_hat, labels) * lam + criterion(y_f_hat, labels_shuffel) * (1 - lam)
                    else:
                        cost = criterion(y_f_hat, labels)

                l_f = torch.sum(cost * weight)
        optimizer.zero_grad()
        l_f.backward()
        if args.clipping_norm > 0:
            nn.utils.clip_grad_norm_(net.parameters(), args.clipping_norm, norm_type=2)
        optimizer.step()
        train_loss.update(l_f.item(), image.size(0))
        train_top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0 and is_main_process():
            print('At epoch: {:03d} Step: {:03d}/{:03d} AVERAGE Acc: {:.4f} AVERAGE TRAIN loss : {:.4f}'.
                  format(epoch, i, len(trainloader), train_top1.avg, train_loss.val))
    if args.distribute:
        train_loss.synchronize_between_processes()
        train_top1.synchronize_between_processes()

        if is_main_process():
            writer.add_scalar('Train/loss', train_loss.avg, epoch)
            writer.add_scalar('Train/lr', optimizer.param_groups[-1]['lr'], epoch)
            print('At epoch: {:03d} the learning rate is : {:.4f}'.format(epoch, optimizer.param_groups[0]['lr']))
            print('At epoch: {:03d} AVERAGE TRAIN loss : {:.4f}'.format(epoch, train_loss.avg))
    else:
        writer.add_scalar('Train/loss', train_loss.avg, epoch)
        writer.add_scalar('Train/lr', optimizer.param_groups[-1]['lr'], epoch)
        print('At epoch: {:03d} the learning rate is : {:.4f}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('At epoch: {:03d} AVERAGE TRAIN loss : {:.4f}'.format(epoch, train_loss.avg))


def val(epoch, testloader,  net, writer, args):
    net.eval()
    test_loss = AverageMeter()
    top1 =AverageMeter()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs = torch.autograd.Variable(inputs.cuda())
            targets = torch.autograd.Variable(targets.cuda())
            torch.cuda.synchronize()
            outputs = net(inputs)
            test_loss_criterion = nn.CrossEntropyLoss().cuda()
            loss = test_loss_criterion(outputs, targets)
            torch.cuda.synchronize()
        
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            test_loss.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))


    if args.distribute:

        top1.synchronize_between_processes()
        test_loss.synchronize_between_processes()

        if is_main_process():
            writer.add_scalar('Test/loss', test_loss.avg, epoch)
            writer.add_scalar('Test/acc', top1.avg, epoch)
            writer.add_scalar('Test/best_acc', best_acc, epoch)


    else:

        writer.add_scalar('Test/loss', test_loss.avg, epoch)
        writer.add_scalar('Test/acc', top1.avg, epoch)
        writer.add_scalar('Test/best_acc', best_acc, epoch)
    return top1.avg, test_loss





if __name__ == '__main__':
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.distribute:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://'
        )
        args.gpuid = torch.device(f'cuda:{args.local_rank}')
        args.seed = args.local_rank
    else:
        args.gpuid = torch.device(f'cuda:{args.gpuid}')

    best = []
    for i in range(args.multi_runs):
        args.seed = i
        best.append(main(args))
    print(best)