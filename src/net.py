
import torch
import torch.nn as nn
from collections import OrderedDict
import math
from net_sphere import AngleLinear
from cosloss import CosLinear
from cocoloss import CocoLinear,mCocoLinear


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, planes, stride=1):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(planes, planes)
        m['relu1'] = nn.PReLU(planes)
        m['conv2'] = conv3x3(planes, planes)
        m['relu2'] = nn.PReLU(planes)
        self.group1 = nn.Sequential(m)

    def forward(self, x):
        residual = x
        out = self.group1(x) + residual
        return out


class SphereNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,linear='linear',scale=15):
        super(SphereNet, self).__init__()


        self.layer1 = self._make_layer(block, 3, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.fc5 = nn.Linear(512*7*6,512)
        self.drop = nn.Dropout(p=0.5)
        # self.ip2 = nn.Linear(2, 10,bias=False)
        if linear=='AngleLinear':
            self.ip = AngleLinear(512, num_classes)
            print("Use AngleLinear")
        elif linear=='CosLinear':
            self.ip = CosLinear(512,num_classes)
            print("Use CosLinear")
        elif linear=='CocoLinear':
            self.ip = CocoLinear(512,num_classes,c=scale)
            print("Use CocoLinear")
        elif linear=='mCocoLinear':
            self.ip = mCocoLinear(512,num_classes,c=scale)
            print("Use CocoLinear")
        else:
            self.ip = nn.Linear(512, num_classes,bias=False)
        # self.ip2 = CosLinear(2, 10)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


    def _make_layer(self, block, inplanes, planes, blocks, stride=2):

        layers = []
        layers.append(nn.Sequential(
                conv3x3(inplanes, planes, stride=stride),
                nn.PReLU(planes))
                )

        for _ in range(0, blocks):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.group1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0),-1)
        x=self.drop(x)
        feat = self.fc5(x)
        out = self.ip(feat)

        return  feat,out

def sphere64a(linear="linear",scale=15,num_classes=79077):
    return SphereNet(BasicBlock, [3,8,16,3],num_classes=num_classes,linear='linear',scale=15)
