from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
import higher
import dataloader_cifar as dataloader
import time
import datetime

from models.resnet import SupCEResNet

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--single_meta', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_file', default=None, type=str)

parser.add_argument('--log_wandb', default=False, type=bool)
parser.add_argument('--wandb_project', default='', type=str)
parser.add_argument('--wandb_experiment', default='', type=str)
parser.add_argument('--wandb_entity', default='',type=str)
parser.add_argument('--wandb_resume', default=False, type=bool)
parser.add_argument('--need_clean', default=False, type=bool)
parser.add_argument('--method', default=None, type=str)
parser.add_argument('--net', default=None, type=str)
parser.add_argument('--warmup', default=10, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

try:
    import wandb
    has_wandb = True
   # os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
except ImportError:
    has_wandb = False
    print('please install wandb')
if args.log_wandb:
    if has_wandb:
        wandb.init(project=str(args.wandb_project), name=str(args.wandb_experiment),
                       entity=str(args.wandb_entity), resume=args.wandb_resume)
        wandb.config.update(args)


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, meta_loader=None):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//(args.batch_size * 2))+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if meta_loader is not None:
            net = l2b(net, optimizer, input_a, target_a, metaloader=meta_loader)

           # net = l2b(net, optimizer, input_a, target_a, metaloader=labeled_trainloader) # using train set
        
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()
        log_metric = {
            'dataset': args.dataset,
             'noise_ratio': args.r,
              'laebled_loss':Lx.item(),
            'unlabled_loss':Lu.item()
        }
        if has_wandb and args.log_wandb:
                wandb.log(log_metric)



def l2b(net, optimizer, image, labels, metaloader):
        criterion =  nn.CrossEntropyLoss(reduction="none").cuda()
        with higher.innerloop_ctx(net, optimizer) as (meta_net, meta_optimizer):
            for s in range(1):
                y_f_hat = meta_net(image)
                _, pesudo_labels = y_f_hat.max(1)
                cost1 = criterion(y_f_hat, labels)
                cost2 = criterion(y_f_hat, pesudo_labels)


                cost = torch.cat((cost1.unsqueeze(0), cost2.unsqueeze(0)), dim=0)

                eps = torch.zeros(cost.size()).cuda()
                eps = eps.requires_grad_()

                l_f_meta = torch.sum(cost * eps)

                meta_optimizer.step(l_f_meta)

            #
            if args.need_clean:
                val_data, val_labels = next(iter(metaloader))
            else:
                val_data, _, val_labels, _ = next(iter(metaloader))
            val_data = val_data.cuda()
            val_labels = val_labels.cuda()

            y_g_hat = meta_net(val_data)
            l_g_meta = torch.mean(criterion(y_g_hat, val_labels))

            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0].detach()

            w_tilde = -grad_eps
            w_tilde = w_tilde.view(-1)

            w_tilde = torch.clamp(w_tilde, min=0)

            norm_c = torch.sum(w_tilde) + 1e-10

            if norm_c != 0:
                weight = w_tilde / norm_c
            else:
                weight = w_tilde

            weight = weight.view(2, -1)

        y_f_hat = net(image)
        _, pesudo_labels = y_f_hat.max(1)

        cost1 = criterion(y_f_hat, labels)
        cost2 = criterion(y_f_hat, pesudo_labels)

        cost = torch.cat((cost1.unsqueeze(0), cost2.unsqueeze(0)), dim=0)

        l_f = torch.sum(cost * weight)

        optimizer.zero_grad()
        l_f.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.80, norm_type=2)
        optimizer.step()

        return net



def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    if has_wandb:
        if args.log_wandb:
            wandb.log({'test_acc':acc})
    test_log.flush()  

def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

stats_log=open('/home/ec2-user/disk/checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w')
test_log=open('/home/ec2-user/disk/checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')

if args.dataset=='cifar10':
    warm_up = args.warmup
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=stats_log,noise_file=args.noise_file)
import torchvision.transforms as transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
meta_loader= dataloader.cifar_dataset(args.dataset,r=args.r,root_dir=args.data_path,noise_mode=args.noise_mode,
                                      transform=transform_train, mode='meta', noise_file=args.noise_file)
from torch.utils.data import Dataset, DataLoader
meta_loader = DataLoader(
                dataset=meta_loader,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=5)
#meta_loader=None
print('| Building net')
def create_model_selfsup(net='resnet18', dataset='cifar10', num_classes=10, device='cuda:0', drop=0):
    chekpoint = torch.load('/home/ec2-user/disk/pretrained/pretrained/ckpt_{}_{}.pth'.format(dataset, net))
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = SupCEResNet(net, num_classes=num_classes)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    return model
if args.method == 'selfsup':
    print('load simclr weitghes')
    net1 = create_model_selfsup(args.net)
    net2 = create_model_selfsup(args.net)
else:
    net1 = create_model()
    net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    start_time = time.time()
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader = loader.run('eval_train')   
    
    if epoch<warm_up:       
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader) 
   
    else:         
        prob1,all_loss[0]=eval_train(net1,all_loss[0])   
        prob2,all_loss[1]=eval_train(net2,all_loss[1])          
               
        pred1 = (prob1 > args.p_threshold)      
        pred2 = (prob2 > args.p_threshold)      
        
        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2,  prob2) # co-divide

        if args.single_meta<1:
            second_meta = None
            first_meta = None
        elif args.single_meta==1:
            second_meta = None
            if args.need_clean:
                first_meta = meta_loader
            else:
                first_meta = labeled_trainloader


        train(epoch,net1,net2,optimizer1, labeled_trainloader, unlabeled_trainloader, meta_loader=first_meta) # train net1
        
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide

        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader, meta_loader=second_meta) # train net2
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    test(epoch,net1,net2)  


