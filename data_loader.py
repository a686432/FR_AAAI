import os, shutil
import numpy as np
import torch, torch.nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import scipy.io as scio

class MyDataSet(Dataset):
    def __init__(self, root="../", listdir="", transform=torchvision.transforms.ToTensor(), target_transform=None):
        self.root = root
        self.data = []
        self.transform = transform
        self.target_transform = target_transform
        self.catalog = root 
        self.listdir = listdir

        if not os.path.exists(self.catalog):
            raise ValueError("Cannot find the data!")

        with open(listdir,"r") as f:
            for line in f:
                imgname,faceid=line.split(" ")
                imgname=os.path.join(self.catalog,imgname)
                label=int(faceid)
                self.data.append([imgname, label])

                

                
    def __getitem__(self, index):
        imgname, label = self.data[index]
        img = cv2.imread(imgname)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label

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
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(trainloader))
    inputdata, target = dataset.__getitem__(2)
    print(inputdata)
    print(target)


if __name__ == "__main__":
    Test()