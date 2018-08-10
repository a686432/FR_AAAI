import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)


class ArcLinear(nn.Module):
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


class ArcLoss(nn.Module):
    def __init__(self, num_cls=10, s=15, alpha=0.1):
        super(ArcLoss, self).__init__()
        self.num_cls = num_cls
        self.alpha = alpha
        self.scale = 0.5*math.log(num_cls-1)+s*math.log(10)/2

    def forward(self, feat, y):
        y = y.view(-1, 1)
        batch_size = feat.size()[0]
        sint=torch.sqrt(1-feat*feat)
        sinm=math.sin(self.alpha)
        cosm=math.cos(self.alpha)
        margin_xw_norm = feat*cosm - sinm*sint
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