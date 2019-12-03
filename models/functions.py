from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def train(net, args, optimizer, train_loader_source, train_loader_target, known_labels):
    net.train()
    # Losses
    domain_loss = nn.NLLLoss()
    classification_loss = nn.NLLLoss()

    # Training
    for epoch in range(args.n_epochs):

        len_dataloader = min(len(train_loader_source), len(train_loader_target))
        data_source_iter = iter(train_loader_source)
        data_target_iter = iter(train_loader_target)

        for i in range(0, len_dataloader, args.batch_size):

            
            p = float(i + epoch * len_dataloader) / args.n_epochs / len_dataloader
            #alpha = 2. / (1. + np.exp(-10 * p)) - 1
            alpha=0.1

            # Training using source data
            s_img, s_label, _ = data_source_iter.next()
            s_img, s_label = s_img.type(torch.FloatTensor), s_label.type(torch.LongTensor)
            domain_label_source = torch.ones(args.batch_size, dtype=torch.long)
            if args.cuda:
                s_img, s_label = s_img.cuda(), s_label.cuda()
                domain_label_source = domain_label_source.cuda()

            net.zero_grad()

            class_output, domain_output = net(s_img, args.alpha)
            error_s_class = classification_loss(class_output, s_label)
            error_s_domain = domain_loss(domain_output, domain_label_source)


            
            # Training using target data
            
            t_img, t_label, t_idx = data_target_iter.next()
            t_img, t_label, t_idx = t_img.type(torch.FloatTensor), t_label.type(torch.LongTensor),t_idx.type(torch.LongTensor).cpu().numpy()
            domain_label_target = torch.zeros(args.batch_size, dtype=torch.long)
            if args.cuda:
                t_img, t_label = t_img.cuda(), t_label.cuda()
                domain_label_target = domain_label_target.cuda()

            class_output_target, domain_output_target = net(t_img, args.alpha)
            if known_labels and (np.isin(t_idx,known_labels)==True).any():
                error_t_class = classification_loss(class_output_target[np.isin(t_idx,known_labels)] , t_label[np.isin(t_idx,known_labels)])
            else:
                error_t_class = 0
            error_t_domain = domain_loss(domain_output_target, domain_label_target)


            
            # Final Loss
            loss = error_s_class + error_s_domain + error_t_class + error_t_domain

            loss.backward()
            optimizer.step()

        # Show statistics
        if args.verbose:
            print('Epoch: %d, [i: %d / %d] & Losses : Cs=%f, Ds=%f, Dt: %f' \
              % (epoch, i, len_dataloader, error_s_class.cpu().data.numpy(),
                 error_s_domain.cpu().data.numpy(), error_t_domain.cpu().data.numpy()))

    return net


def test(net, args, test_loader_target):

    net.eval()

    nb_correct_classification = 0
    nb_samples = 0
    nb_correct_domain = 0

    for t_img, t_label, _ in test_loader_target:
        t_img, t_label= t_img.type(torch.FloatTensor), t_label.type(torch.LongTensor)
        if args.cuda:
            t_img, t_label = t_img.cuda(), t_label.cuda()
            
        with torch.no_grad():
            class_output, domain_output = net(t_img, args.alpha)
        prediction = torch.argmax(class_output, dim=1)
        domain_pred = torch.argmax(domain_output, dim=1)

        nb_correct_classification += torch.sum(prediction.eq(t_label))
        nb_correct_domain += torch.sum(domain_pred==0)
        nb_samples += args.batch_size

    accuracy = nb_correct_classification.cpu().numpy() / nb_samples
    dom_accuracy = nb_correct_domain.cpu().numpy() / nb_samples
    print('\t Ac: %f & Ad: %f' % (accuracy, dom_accuracy))


def score(model, args, train_loader_target, known_labels):
    model.eval()
    # p = look in original paper for clues
    # alpha = look in original paper for clues
    s = False
    for data, target, idx in train_loader_target:
        data, target, idx = data.type(torch.FloatTensor), target.type(torch.LongTensor), idx.type(torch.LongTensor).cpu().numpy()
        idx_filter = np.logical_not(np.isin(idx, known_labels))
        if (idx_filter==False).all():
            continue
        data, target = data[idx_filter], target[idx_filter]
        if args.cuda:
            data = data.cuda()
            target = target.cuda()
        with torch.no_grad():
            cf,df = model(data,args.alpha)
        df = df[:,1] / df.sum(dim=1)
        
        w = (1 - df) / df  # Importance weight
        
        H = lambda x : -1*torch.sum(torch.exp(x)* x, dim=1)
        if s is False:
            s = w * H(cf)
            s = s.cpu()
            idxs = idx
        else:
            s2 = w * H(cf)
            s = torch.cat((s,s2.cpu()))
            idxs = np.concatenate((idxs, idx))

    return idxs, s.cpu().numpy()
