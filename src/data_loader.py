import os, shutil, sys
import numpy as np
import torch, torch.nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import scipy.io as scio

class MyDataSet(Dataset):
    def __init__(self, max_number_class=100000,root="../", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.catalog = root 

        if not os.path.exists(self.catalog):
            raise ValueError("Cannot find the data!")

        dirs = os.listdir(self.catalog)
        i=0
        for dir in dirs:
            i+=1
            sys.stdout.write(str(i)+'/'+str(len(dirs))+'\r')
            imgs = os.listdir(self.catalog + dir)
            label = int(dir)
            if label<max_number_class:
                for img in imgs:
                    imgname = self.catalog + dir + "/" + img         
                    self.data.append([imgname, label])
                
    def __getitem__(self, index):
        imgname, label = self.data[index]
        
        img = cv2.imread(imgname)
        if img is None:
            print(imgname)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

class LfwDataSet(Dataset):
    def __init__(self, root="../", pairfile="../pair.txt", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.pairfile = pairfile 

        if not os.path.exists(self.root):
            raise ValueError("Cannot find the data!")
        with open('../pairs.txt') as f:
            pairs_lines = f.readlines()[1:]
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')
            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            if 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            imgname1 = self.root + name1 
            imgname2 = self.root + name2 
            self.data.append([imgname1,imgname2,sameflag])
           

                
    def __getitem__(self, index):
        imgname1, imgname2, label = self.data[index]
        img1 = cv2.imread(imgname1)
        img2 = cv2.imread(imgname2)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2,label

    def __len__(self):
        return len(self.data)

def Test():
    print("For test.\n")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor()
    ])
    dataset = MyDataSet(transform=transform, root="/data3/jdq/imgc/")
    #dataset = MyDataSet(transform=transform, root="./output/")
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=32)
    print(len(trainloader))
    inputdata, target = dataset.__getitem__(2)
    print(inputdata)
    print(target)


if __name__ == "__main__":
    Test()