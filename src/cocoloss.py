import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math

class CocoLinear(nn.Module):
    """
        Refer to paper:
        Yu Liu, Hongyang Li, Xiaogang Wang
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, feat_dim, num_classes, c=21):
        super(CocoLinear, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        #self.alpha = 0.5*math.log(num_classes-1)+c*math.log(10)/2
        self.alpha = c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha*nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))
        # loss = nn.CrossEntropyLoss()
        # output = loss(logits, y)

        return logits

class mCocoLinear(nn.Module):
    """
        Refer to paper:
        Yu Liu, Hongyang Li, Xiaogang Wang
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, feat_dim, num_classes, c=21):
        super(mCocoLinear, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        #self.alpha = 0.5*math.log(num_classes-1)+c*math.log(10)/2
        self.alpha = c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.phi=nn.Parameter(torch.Tensor(1))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))
        logits = self.alpha*(logits+self.phi.expand_as(logits))
        # loss = nn.CrossEntropyLoss()
        # output = loss(logits, y)

        return logits


class CocoLoss(nn.Module):
    def __init__(self):
        super(CocoLoss, self).__init__()

    def forward(self, feat, y):
        loss = nn.CrossEntropyLoss()
        output = loss(feat, y)
        return output