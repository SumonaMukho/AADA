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
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # Training using source data
            input_source = data_source_iter.next()
            s_img, s_label, _ = input_source
            s_img, s_label = s_img.type(torch.FloatTensor), s_label.type(torch.LongTensor)
            domain_label_source = torch.ones(args.batch_size, dtype=torch.long)
            if args.cuda:
                s_img, s_label = s_img.cuda(), s_label.cuda()
                domain_label_source = domain_label_source.cuda()

            net.zero_grad()

            class_output, domain_output = net(s_img, alpha)
            error_s_class = classification_loss(class_output, s_label)
            error_s_domain = domain_loss(domain_output, domain_label_source)


            
            # Training using target data
            
            input_target = data_target_iter.next()
            t_img, t_label, t_idx = input_target
            t_img, t_label, t_idx = t_img.type(torch.FloatTensor), t_label.type(torch.LongTensor),t_idx.type(torch.LongTensor).cpu().numpy()
            domain_label_target = torch.zeros(args.batch_size, dtype=torch.long)
            if args.cuda:
                t_img, t_label = t_img.cuda(), t_label.cuda()
                domain_label_target = domain_label_target.cuda()

            class_output_target, domain_output_target = net(t_img, alpha)
            if known_labels:
                error_t_class = classification_loss(class_output_target[np.isin(t_idx,known_labels)] , t_label[np.isin(t_idx,known_labels)])
                print("nb known labels",t_label[np.isin(t_idx,known_labels)].shape)
            else:
                error_t_class = 0
            error_t_domain = domain_loss(domain_output_target, domain_label_target)


            
            # Final Loss
            loss = error_s_class + error_s_domain + error_t_class + error_t_domain

            loss.backward()
            optimizer.step()

        # Show statistics
        print('Epoch: %d, [i: %d / %d] & Losses : Cs=%f, Ds=%f, Dt: %f' \
              % (epoch, i, len_dataloader, error_s_class.cpu().data.numpy(),
                 error_s_domain.cpu().data.numpy(), error_t_domain.cpu().data.numpy()))

    return net


def test(net, args, test_loader_target):

    net.eval()

    nb_correct_classification = 0
    nb_samples = 0
    nb_correct_domain = 0

    for _, (t_img, t_label) in enumerate(test_loader_target):
        t_img, t_label= t_img.type(torch.FloatTensor), t_label.type(torch.LongTensor)
        if args.cuda:
            t_img, t_label = t_img.cuda(), t_label.cuda()

        class_output, domain_output, _ = net(t_img, 0)
        prediction = torch.argmax(class_output, dim=1)
        domain_pred = torch.argmax(domain_output, dim=1)

        nb_correct_classification += torch.sum(prediction.eq(t_label))
        nb_correct_domain += torch.sum(domain_pred==0)
        nb_samples += args.batch_size

    accuracy = nb_correct_classification.numpy() / nb_samples
    dom_accuracy = nb_correct_domain.numpy() / nb_samples
    print('accuracy in label classif test: %f' % (accuracy))
    print('accuracy in domain classif test: %f' % (dom_accuracy))


def score(model, args, data, target):
    model.eval()
    # p = look in original paper for clues
    # alpha = look in original paper for clues
    data, target = data.type(torch.FloatTensor), target.type(torch.LongTensor)
    if args.cuda:
        data = data.cuda()
        target = target.cuda()
    cf,df = model(data,target)
    
    df = df[:,0] / df.sum(dim=1)
    
    w = (1 - df) / df  # Importance weight
    
    H = lambda x : torch.sum((-1 * torch.mm(F.softmax(x, dim=1), F.log_softmax(x, dim=1).transpose(0, 1))), dim=1)
    s = w * H(cf)
    return s
