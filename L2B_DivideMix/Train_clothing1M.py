from __future__ import print_function
import sys

import higher
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
import argparse
import numpy as np
import dataloader_clothing1M as dataloader
from sklearn.mixture import GaussianMixture

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=80, type=int)
parser.add_argument('--id', default='clothing1m')
parser.add_argument('--data_path', default='../../Clothing1M/data', type=str, help='path to dataset')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=14, type=int)
parser.add_argument('--num_batches', default=1000, type=int)


parser.add_argument('--log_wandb', default=False, type=bool)
parser.add_argument('--wandb_project', default='', type=str)
parser.add_argument('--wandb_experiment', default='', type=str)
parser.add_argument('--wandb_entity', default='xianhangli',type=str)
parser.add_argument('--wandb_resume', default=False, type=bool)
parser.add_argument('--need_clean', default=False, type=bool)
parser.add_argument('--single_meta', default=0, type=int)
parser.add_argument('--moco_pretrained', default=None, type=str)
parser.add_argument('--warmup', default=1, type=int)
parser.add_argument('--cos_lr', default=False, type=bool)

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


# l2b module
def l2b(net, optimizer, image, labels, metaloader):
        criterion = nn.CrossEntropyLoss(reduction="none").cuda()
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
        #nn.utils.clip_grad_norm_(net.parameters(), 0.80, norm_type=2)
        optimizer.step()

        return net


# Training
def train(epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader, meta_loader=None):
    net.train()
    net2.eval() #fix one network and train the other
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
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
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if meta_loader is not None:
            net = l2b(net, optimizer, input_a,  target_a, metaloader=meta_loader)

        sys.stdout.write('\r')
        sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()
        log_metric = {
            'laebled_loss': Lx.item(),
        }
        if has_wandb and args.log_wandb:
            wandb.log(log_metric)
    
def warmup(net,optimizer,dataloader):
    net.train()
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty       
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, args.num_batches, loss.item(), penalty.item()))
        sys.stdout.flush()
    
def val(net,val_loader,k):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    print("\n| Test\t Net%d  Acc: %.2f%%" %(k,acc))
    if acc > best_acc[k-1]:
        best_acc[k-1] = acc
        print('| Saving Best Net%d ...'%k)
        save_point = '/home/ec2-user/disk/checkpoint/%s_net%d.pth.tar'%(args.id,k)
        torch.save(net.state_dict(), save_point)
    if has_wandb:
        if args.log_wandb:
            wandb.log({'eval_acc':acc})
    return acc

def test(net1,net2,test_loader):
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
    print("\n| Test Acc: %.2f%%\n" %(acc))
    if has_wandb:
        if args.log_wandb:
            wandb.log({'test_acc': acc})
    return acc    
    
def eval_train(epoch,model):
    model.eval()
    num_samples = args.num_batches*args.batch_size
    losses = torch.zeros(num_samples)
    paths = []
    n=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[n]=loss[b] 
                paths.append(path[b])
                n+=1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob,paths  
    
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
               
def create_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048,args.num_class)
    if args.moco_pretrained:
        if os.path.isfile(args.moco_pretrained):
            print("=> loading checkpoint '{}'".format(args.moco_pretrained))
            checkpoint = torch.load(args.moco_pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.moco_pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.moco_pretrained))
    model = model.cuda()
    return model     

log=open('/home/ec2-user/disk/checkpoint/%s.txt'%args.id,'w')
log.flush()

loader = dataloader.clothing_dataloader(root=args.data_path,batch_size=args.batch_size,num_workers=5,num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
                      
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

import torchvision.transforms as transforms
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
])
meta_loader= dataloader.clothing_dataset(args.data_path, transform=transform_train, mode='meta')
from torch.utils.data import Dataset, DataLoader
meta_loader = DataLoader(
                dataset=meta_loader,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=5)

best_acc = [0,0]
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.num_epochs)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.num_epochs)

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if args.cos_lr:
        scheduler1.step()
        scheduler2.step()
    else:
        if epoch >= 40:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
        
    if epoch<args.warmup:     # warm up
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1,optimizer1,train_loader)     
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2,optimizer2,train_loader)                  
    else:       
        pred1 = (prob1 > args.p_threshold)  # divide dataset  
        pred2 = (prob2 > args.p_threshold)      
        
        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2,paths=paths2) # co-divide


        if args.single_meta<1:
            second_meta = None
            first_meta = None
        elif args.single_meta==1:
            second_meta = None
            if args.need_clean:
                first_meta = meta_loader
            else:
                first_meta = labeled_trainloader

        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader,meta_loader=first_meta)              # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1,paths=paths1) # co-divide
        train(epoch,net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader,meta_loader=second_meta)              # train net2
    
    val_loader = loader.run('test') # validation
    acc1 = val(net1,val_loader,1)
    acc2 = val(net2,val_loader,2)   
    log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
    log.flush() 
    print('\n==== net 1 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch  
    prob1,paths1 = eval_train(epoch,net1) 
    print('\n==== net 2 evaluate next epoch training data loss ====') 
    eval_loader = loader.run('eval_train')  
    prob2,paths2 = eval_train(epoch,net2) 

test_loader = loader.run('test')
net1.load_state_dict(torch.load('/home/ec2-user/disk/checkpoint/%s_net1.pth.tar'%args.id))
net2.load_state_dict(torch.load('/home/ec2-user/disk/checkpoint/%s_net2.pth.tar'%args.id))
acc = test(net1,net2,test_loader)      

log.write('Test Accuracy:%.2f\n'%(acc))
log.flush() 
