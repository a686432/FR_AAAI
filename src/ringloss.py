import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import numpy as np

class RingLoss(nn.Module):
    def __init__(self, type='auto', loss_weight=1.0):
        """
        :param type: type of loss ('l1', 'l2', 'auto')
        :param loss_weight: weight of loss, for 'l1' and 'l2', try with 0.01. For 'auto', try with 1.0.
        :return:
        """
        super(RingLoss, self).__init__()
        self.radius = Parameter(torch.Tensor(1))
        self.radius.data.fill_(-1)
        self.loss_weight = loss_weight
        self.type = type

    def forward(self, x):
        x = x.pow(2).sum(dim=1).pow(0.5)
        if self.radius.data[0] < 0: # Initialize the radius with the mean feature norm of first iteration
            self.radius.data.fill_(x.mean().data)
        if self.type == 'l1': # Smooth L1 Loss
            loss1 = F.smooth_l1_loss(x, self.radius.expand_as(x)).mul_(self.loss_weight)
            loss2 = F.smooth_l1_loss(self.radius.expand_as(x), x).mul_(self.loss_weight)
            ringloss = loss1 + loss2
        elif self.type == 'auto': # Divide the L2 Loss by the feature's own norm
            diff = x.sub(self.radius.expand_as(x)) / (x.mean().detach().clamp(min=0.5))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        else: # L2 Loss, if not specified
            diff = x.sub(self.radius.expand_as(x))
            diff_sq = torch.pow(torch.abs(diff), 2).mean()
            ringloss = diff_sq.mul_(self.loss_weight)
        return ringloss