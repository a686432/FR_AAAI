import torch
import torch.nn as nn
from torch.autograd.function import Function

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True, loss_weight=1):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average
        self.loss_weight=loss_weight

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        loss = self.centerlossfunc(feat, label, self.centers)
        loss /= (batch_size if self.size_average else 1)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff, None, grad_centers



