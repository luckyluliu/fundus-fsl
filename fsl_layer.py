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
    dists = euclidean_dist(input_cpu, prototypes)
    #print(dists.size(), input_cpu.size(), prototypes.size())
    log_p_y = F.log_softmax(-dists, dim=1)
    #print(log_p_y.size())
    _, y_hat = log_p_y.max(1)
    #print(y_hat.size(), target_cpu.size())
    acc_val = y_hat.eq(target_cpu).float().mean()
    return acc_val, y_hat
