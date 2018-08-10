import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)


class CosLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features,in_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input):
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        output = cos_theta.clamp(-1, 1)

        return output


class CosLoss(nn.Module):
    def __init__(self, num_cls=10, s=15, alpha=0.1):
        super(CosLoss, self).__init__()
        self.num_cls = num_cls
        self.alpha = alpha
        self.scale = s
        self.phi=nn.Parameter(torch.Tensor(1))
        self.phi.data.uniform_(s, -s)


    def forward(self, feat, y):
        y = y.view(-1, 1)
        batch_size = feat.size()[0]
        feat = feat + self.phi.expand_as(feat)
        margin_xw_norm = feat - self.alpha
        y_onehot = torch.Tensor(batch_size, self.num_cls).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, y.data.view(-1, 1), 1)
        y_onehot.byte()
        y_onehot = Variable(y_onehot)
        value = self.scale * where(y_onehot, margin_xw_norm, feat)
        # value = value
        # logpt = F.log_softmax(value)
        y = y.view(-1)
        loss = nn.CrossEntropyLoss()
        output = loss(value, y)
        # loss = loss.mean()
        return output

class mCosLoss(nn.Module):
    def __init__(self, num_cls=10, s=15, alpha=0.1):
        super(mCosLoss, self).__init__()
        self.num_cls = num_cls
        self.alpha = alpha
        self.scale = s
        self.phi=nn.Parameter(torch.Tensor(1))

    def forward(self, feat, y):
        y = y.view(-1, 1)
        batch_size = feat.size()[0]
        feat = feat + self.phi.expand_as(feat)
        margin_xw_norm_h = feat - self.alpha 
        margin_xw_norm_l = feat + self.alpha
        y_onehot = torch.Tensor(batch_size, self.num_cls).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, y.data.view(-1, 1), 1)
        y_onehot.byte()
        y_onehot = Variable(y_onehot)
        value = self.scale * where(y_onehot, margin_xw_norm_h, margin_xw_norm_l)
        # value = value
        # logpt = F.log_softmax(value)
        y = y.view(-1)
        loss = nn.CrossEntropyLoss()
        output = loss(value, y)
        # loss = loss.mean()
        return output