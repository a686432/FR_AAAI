import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from resnet import ResNet18
from CenterLoss import CenterLoss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import argparse
from utils import progress_bar
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser(description='Depth img with CenterLoss')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lrc', default=0.5, type=float, help="lr for center loss")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--verify', action='store_true', help='Verify the net')
parser.add_argument('--gpu', default="0", help="select the gpu")
parser.add_argument('--w', default=0.01,type=float, help="weight for center loss")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# Dataset
trainset = datasets.MNIST('../MNIST', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=4)

testset = datasets.MNIST('../MNIST', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

def visualize(feat, labels, epoch, train=True):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    # plt.xlim(xmin=-30,xmax=30)
    # plt.ylim(ymin=-30,ymax=30)
    plt.text(-14.8,14.6,"epoch=%d" % epoch,)
    if train:
        plt.savefig('../images/epoch=%d.jpg' % epoch)
    else:
        plt.savefig('../images/test_epoch=%d.jpg' % epoch)
    # plt.draw()
    # plt.pause(0.001)


def train(epoch):
    print "Training... Epoch = %d" % epoch
    ip1_loader = []
    idx_loader = []
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        ip1, pred = model(data)
        loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)

        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()

        loss.backward()

        optimizer4nn.step()
        optimzer4center.step()

        ip1_loader.append(ip1)
        idx_loader.append((target))

        train_loss += loss.item()
        _, predicted = pred.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch) 
def test(epoch):
    print "Test... Epoch = %d" % epoch
    ip1_loader = []
    idx_loader = []
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx,(data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        ip1, pred = model(data)
        loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)

        #optimizer4nn.zero_grad()
        #optimzer4center.zero_grad()

        #loss.backward()

        #optimizer4nn.step()
        #optimzer4center.step()

        ip1_loader.append(ip1.data)
        idx_loader.append((target.data))

        test_loss += loss.item()
        _, predicted = pred.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch,train=False)




# Model
print('==> Building model..')
model = ResNet18().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
# NLLLoss
nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
loss_weight = 0.01
centerloss = CenterLoss(10, 2).to(device)

# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

# optimzer4center
optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)

for epoch in range(100):
    sheduler.step()
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1)
    test(epoch+1)


