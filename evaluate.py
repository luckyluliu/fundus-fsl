# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_eval(input, target, prototypes):
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    '''
    # 找出c类的所有样本，然后提取出前n_support个
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = 8
    
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    # idx_list是某一类的support样本，取均值然后叠加，得到的prototypes就是每类的聚类中心的叠加
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]
    
    dists = euclidean_dist(input_cpu, prototypes)
    #print(dists.size(), input_cpu.size(), prototypes.size(), n_classes, n_query)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    #print(target_inds.size())
    target_inds = target_inds.view(n_classes, 1, 1)
    
    target_inds.size()
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    #print(log_p_y.size(), target_inds.size())
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    '''
    dists = euclidean_dist(input_cpu, prototypes)
    #print(dists.size(), input_cpu.size(), prototypes.size())
    log_p_y = F.log_softmax(-dists, dim=1)
    #print(log_p_y.size())
    _, y_hat = log_p_y.max(1)
    #print(y_hat.size(), target_cpu.size())
    acc_val = y_hat.eq(target_cpu).float().mean()

    return acc_val
