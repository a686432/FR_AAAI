import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
#from resnet import ResNet18,ResNet34
from CenterLoss import CenterLoss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import argparse
#from utils import progress_bar
import torch.backends.cudnn as cudnn
import lfw
from net_sphere import AngleLoss, sphere20a
from ringloss import RingLoss
from cosloss import CosLoss, mCosLoss
from cocoloss import CocoLoss
from data_loader import MyDataSet,LfwDataSet
import numpy as np
from resnet import resnet50
from net import sphere64a



parser = argparse.ArgumentParser(description='Face recognition with CenterLoss')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lrc', default=0.5, type=float, help="lr for center loss")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--verify', action='store_true', help='Verify the net')
parser.add_argument('--gpu', default="0", help="select the gpu")
parser.add_argument('--w', default=0.01,type=float, help="weight for loss function")
parser.add_argument('--s', default=15,type=float, help="scale for loss function")
parser.add_argument('--outpath',  default="../images", help='Verify the net')
parser.add_argument('--lossfunc','-lf',default='softmax',help='loss function')
parser.add_argument('--m', default=0,type=float, help="weight for loss function")
parser.add_argument('--number_of_class','-nc', default=79077,type=int, help="The number of the class")
args = parser.parse_args()



transform = transforms.Compose([
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        # transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomResizedCrop(112,scale=(0.5,2)),
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
eval_epoch=0
iter_num=0
Best_Accu=0
savefile = "../dict2.cl"

lr=args.lr
margin=args.m
number_of_class=args.number_of_class
mainloss='softmax'
losstype=args.lossfunc.split('+')

print(losstype)
if len(losstype)==1:
    auxiliaryfunc=False
    if losstype[0] not in ["softmax",'A-softmax','cosloss','arcloss','cocoloss','mcosloss','mcocoloss']:
        print('Bad input of loss!')
        exit(1)
    mainloss=losstype[0]
elif len(losstype)==2:
    auxiliaryfunc=True
    if losstype[1] not in ['centerloss','ringloss']:
        print('Bad input of loss!')
        exit(1)
    mainloss=losstype[0]
    auxiliaryloss=losstype[1]
else:
    print('Too many losses')
    exit(1)

print ('The loss function we use')
for lf in losstype:
    print (lf)

Best_Accu = 0.0
train_loss = 0
correct = 0
total = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
gpu_ids = [0,1,2,3]
torch.cuda.set_device(gpu_ids[0])
loss_weight=args.w
img_outpath=args.outpath
use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# Dataset
# trainset = datasets.MNIST('../MNIST', train=True, transform=transforms.Compose([
#     transforms.Resize((112, 96)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))]))
# train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

# testset = datasets.MNIST('../MNIST', train=False, transform=transforms.Compose([
#     transforms.Resize((112, 96)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))]))
# test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)


trainset = MyDataSet(root="/data3/jdq/imgc/",max_number_class=number_of_class, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=32)

evalset = LfwDataSet(root="/data3/jdq/lfwc/",pairfile="../pair.txt",transform=transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

eval_loader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=16)

# Model
print('==> Building model..')
#model=None
if mainloss=='A-softmax':
    model = sphere64a(linear='AngleLinear',num_classes=number_of_class)
elif mainloss=='cosloss' or mainloss=='mcosloss' or mainloss=='arcloss':
    model = sphere64a(linear='CosLinear',num_classes=number_of_class)
elif mainloss == 'cocoloss':
    model = sphere64a(linear='CocoLinear',scale=args.s,num_classes=number_of_class)
elif mainloss == 'mcocoloss':
    model = sphere64a(linear='mCocoLinear',scale=args.s,num_classes=number_of_class)
else: 
    model = sphere64a(num_classes=number_of_class)

model = model.to(device)
model = torch.nn.DataParallel(model,device_ids=gpu_ids)
cudnn.benchmark = True

# NLLLoss
# nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
# centerloss = CenterLoss(10, 2).to(device)
if mainloss=='A-softmax':
    lossfunction = AngleLoss().to(device)
elif mainloss=='cosloss':
    lossfunction = CosLoss(num_cls=number_of_class,s=args.s, alpha=margin).to(device)
elif mainloss == 'cocoloss' or mainloss=='mcocoloss':
    lossfunction = CocoLoss().to(device)
elif mainloss == 'mcosloss':
    lossfunction = mCosLoss(num_cls=number_of_class,s=args.s, alpha=margin).to(device)
elif mainloss == 'arcloss':
    lossfunction = mCosLoss(num_cls=number_of_class,s=args.s, alpha=margin).to(device)
else:
    lossfunction = nn.CrossEntropyLoss().to(device)
    #lossfunction=nn.DataParallel(lossfunction, device_ids=gpu_ids)

# ringloss = RingLoss(loss_weight=loss_weight).to(device)
# lossfunction = CosLoss(num_cls=10).to(device)
#lossfunction2 = RingLoss(loss_weight=loss_weight).to(device)
lossfunction2= None
criterion = [lossfunction,lossfunction2]
# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=lr,momentum=0.9, weight_decay=0.0005)
optimizer4nn = nn.DataParallel(optimizer4nn, device_ids=gpu_ids)
#optimizer4l = optim.SGD(lossfunction.parameters(),lr=lr,momentum=0.9, weight_decay=0.0005)
# sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)

# optimzer4loss
if auxiliaryfunc==True:
    optimzer4loss = optim.SGD(lossfunction2.parameters(), lr =0.5)



def train(epoch):     
    global lr, eval_epoch, iter_num,train_loss,total,correct
    print ("Training... Epoch = %d" % epoch)
    model.train()
  
    for batch_idx,(data, target) in enumerate(train_loader):
        data, target =  data.to(device), target.to(device)

        ip1, pred = model(data)
        # loss = nllloss(pred, target) + loss_weight * centerloss(target, ip1)
        # loss=criterion[0](pred, target)+criterion[1](ip1)
        
        if auxiliaryfunc:

            loss=criterion[0](pred, target)+criterion[1](ip1)
            optimzer4nn.zero_grad()
            optimzer4loss.zero_grad()
            optimizer4l.zero_grad()
            loss.backward()
            optimizer4nn.module.step()
            optimzer4loss.step()

            optimizer4l.step()
        else:
            loss=criterion[0](pred, target)
            optimizer4nn.module.zero_grad()

            optimizer4nn.zero_grad()

            #optimizer4l.zero_grad()
            loss.backward()
            optimizer4nn.module.step()
            #optimizer4l.step()


        # ip1_loader.append(ip1)
        # idx_loader.append((target))

        train_loss += loss.item()
        if mainloss=="A-softmax":
            _, predicted = pred[0].max(1)
        else:
            _, predicted = pred.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        
        sys.stdout.write("%d/%d Loss: %.3f | Acc: %.3f%% (%d/%d)" 
            % (batch_idx+1,len(train_loader),train_loss/(iter_num%5000+1), 100.*correct/total, correct, total))

        if batch_idx < len(train_loader) - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

        iter_num+=1

        if (iter_num)%5000==0:
            train_loss=0
            total=0
            correct=0
            torch.cuda.empty_cache()
            sys.stdout.write('\n')
            sys.stdout.flush()
            eval_epoch+=1
            eval(eval_epoch)

            # model.eval()
            # model.train()
        if iter_num==15000:
            lr /= 10
            for param_group in optimizer4nn.module.param_groups:
                param_group['lr'] = lr
            print("Modify lr to %.5f" % lr)
        if iter_num==30000:
            lr /= 10
            for param_group in optimizer4nn.module.param_groups:
                param_group['lr'] = lr
            print("Modify lr to %.5f" % lr)
        # feat = torch.cat(ip1_loader, 0)
        # labels = torch.cat(idx_loader, 0)
        #visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)
    


   

def eval(epoch):

    global Best_Accu,lr
    predicts = []
    model.eval()
    for batch_idx,(imagename1,imagename2,data1,data2, target) in enumerate(eval_loader):
        data1, data2, target = data1.to(device),data2.to(device) ,target.to(device)
        ip1, _ = model(data1)
        ip2, _ = model(data2)
        ip1=ip1.data.cpu().numpy()[0]
        ip2=ip2.data.cpu().numpy()[0]
        cosdistance = (ip1.dot(ip2)) / (np.linalg.norm(ip1) * np.linalg.norm(ip2) + 1e-12)
        label=target.data.cpu().numpy()[0]
        predicts.append([imagename1,imagename2,cosdistance,label])
        #print(cosdistance,label)
        sys.stdout.write(str(batch_idx)+'/'+str(len(eval_loader))+'\r')
    accuracy = []
    thd = []
    # folds = lfw.KFold(n=6000, n_folds=10)

    thresholds = np.arange(-1.0, 1.0, 0.005)
    #predicts = np.array(map(lambda line: line.strip('\n').split('\t'), predicts))
    #print(predicts)
    #for idx, (train, test) in enumerate(folds):
        
        #print(train)
    best_thresh = lfw.find_best_threshold(thresholds, predicts)
    accuracy.append(lfw.eval_accd(best_thresh, predicts))
    thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    if (np.mean(accuracy) > Best_Accu):
        state = {
            'dict': (model).state_dict(),
            #'center_loss': center_loss.state_dict(),
            #'best_accu': accuracy,
            #'threshold': Threshold,
            #'delta_threshold': deltaThreshold,
            #'weight': criterion[1].loss_weight,
        }
        print("Saving...\n")
        torch.save(state, savefile)
        Best_Accu = np.mean(accuracy)
        not_save = 0
    else:
        not_save += 1
        # if not_save > 3:
        #     # Threshold += deltaThreshold if fail0 > fail2 else -deltaThreshold
        #     # deltaThreshold *= 0.9
        #     if lr > 0.00001:
        #         lr /= 10
        #     for param_group in optimizer4nn.module.param_groups:
        #         param_group['lr'] = lr
        #     not_save = 0
        #     print("Modify lr to %.5f" % lr)

# def test(epoch):
#     global Best_Accu
#     print ("Test... Epoch = %d" % epoch)
#     ip1_loader = []
#     idx_loader = []
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx,(data, target) in enumerate(test_loader):
#         data, target = data.to(device), target.to(device)

#         ip1, pred = model(data)
#         loss=criterion[0](pred, target)+criterion[1](ip1)

#         ip1_loader.append(ip1.data)
#         idx_loader.append((target.data))

#         test_loss += loss.item()

#         # calculate the accuracy
#         if mainloss=="A-softmax":
#             _, predicted = pred[0].max(1)
#         else:
#             _, predicted = pred.max(1)
#         total += target.size(0)
#         correct += predicted.eq(target).sum().item()

#         # print the result
#         sys.stdout.write("%d/%d Loss: %.3f | Acc: %.3f%% (%d/%d) " \
#             % (batch_idx+1,len(test_loader),test_loss/(batch_idx+1), 100.*correct/total, correct, total))
#         if batch_idx < len(test_loader) - 1:
#             sys.stdout.write('\r')
#         else:
#             sys.stdout.write('\n')
#             if 100.*correct/total>Best_Accu:
#                 Best_Accu=100.*correct/total
#             sys.stdout.write("Bacc:%.3f\n" %(Best_Accu))
#         sys.stdout.flush()
   
#     # visualize
#     if epoch % 10==0:
#         feat = torch.cat(ip1_loader, 0)
#         labels = torch.cat(idx_loader, 0)
#         visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch,train=False)







for epoch in range(500):
    #sheduler.step()
    # print optimizer4nn.param_groups[0]['lr']
    train(epoch+1)
    #eval(epoch+1)

    #test(epoch+1)

