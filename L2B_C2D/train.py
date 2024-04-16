import sys

import numpy as np
import torch
import torch.nn.functional as F


try:
    import wandb

    has_wandb = True
# os.environ["WANDB_API_KEY"] = YOUR_KEY_HERE
except ImportError:
    has_wandb = False
    print('please install wandb')

def co_guess(net, net2, inputs_x, inputs_u, inputs_x2, inputs_u2, w_x, labels_x, T, smooth_clean):
    # label co-guessing of unlabeled samples
    outputs_u11 = net(inputs_u)
    outputs_u12 = net(inputs_u2)
    outputs_u21 = net2(inputs_u)
    outputs_u22 = net2(inputs_u2)

    pu = (torch.softmax(outputs_u11, dim=1) +
          torch.softmax(outputs_u12, dim=1) +
          torch.softmax(outputs_u21, dim=1) +
          torch.softmax(outputs_u22, dim=1)) / 4
    ptu = pu ** (1 / T)  # temperature sharpening

    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
    targets_u = targets_u.detach()

    # label refinement of labeled samples
    outputs_x = net(inputs_x)
    outputs_x2 = net(inputs_x2)

    if smooth_clean:
        px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
        px = w_x * labels_x + (1 - w_x) * px
        ptx = px ** (1 / T)  # temparature sharpening

        targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
        targets_x = targets_x.detach()
    else:
        targets_x = labels_x
    return targets_x, targets_u


import torch.nn as nn
import higher
def l2b(net, optimizer, image,  labels,  metaloader,  args):
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

            if args.need_clean:
                val_data,  val_labels =  next(iter(metaloader))
            else:
                val_data, _, val_labels, _,  _ = next(iter(metaloader))
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


# Training
def train(epoch, net, net2, criterion, optimizer, labeled_trainloader, unlabeled_trainloader, lambda_u, batch_size,
          num_class, device, T, alpha, warm_up, dataset, r, noise_mode, num_epochs, smooth_clean=True, metadata=None, args=None):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, _, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2 = inputs_x.to(device), inputs_x2.to(device)
        labels_x, w_x = labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            targets_x, targets_u = co_guess(net, net2, inputs_x, inputs_u, inputs_x2, inputs_u2, w_x, labels_x, T,
                                            smooth_clean)

        # mixmatch
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        if lambda_u > 0:
            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[:batch_size * 2]
            logits_u = logits[batch_size * 2:]

            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                     epoch + batch_idx / num_iter, warm_up, lambda_u)
        else:
            mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
            mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]

            logits = net(mixed_input)

            Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
            lamb, Lu = 0, 0
        # regularization
        prior = torch.ones(num_class) / num_class
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if metadata:
            #print('tune l2b models')
            if not args.need_clean:
                metadata = labeled_trainloader

            net = l2b(net, optimizer, input_a, target_a, metaloader=metadata, args=args)

        sys.stdout.write('\r')
        if 'cifar' in 'dataset':
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t'
                             'Labeled loss: %.2f  Unlabeled loss: %.2e(%.2e)  penalty: %.2e'
                             % (dataset, r, noise_mode, epoch, num_epochs, batch_idx + 1, num_iter,
                                Lx.item(), Lu.item(), lamb * Lu.item(), penalty.item()))
        elif 'clothing' in dataset:
            sys.stdout.write('Clothing1M | Epoch [%3d/%3d] Iter[%3d/%3d]\t'
                             'Labeled loss: %.2f  penalty: %.2e'
                             % (epoch, num_epochs, batch_idx + 1, num_iter, Lx.item(), penalty.item()))
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader, criterion, conf_penalty, device, dataset, r, num_epochs, noise_mode):
    net.train()
    for batch_idx, (inputs, _, labels, _, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        assert torch.isfinite(loss).all()
        penalty = conf_penalty(outputs) if conf_penalty is not None else 0.
        L = loss + penalty
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()

        sys.stdout.write('\r')
        if 'clothing' in dataset:
            sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                             % (batch_idx + 1, len(dataloader), loss.item(), penalty.item()))
        elif 'cifar' in dataset:
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                             % (dataset, r, noise_mode, epoch, num_epochs, batch_idx + 1, len(dataloader),
                                loss.item()))
        sys.stdout.flush()
