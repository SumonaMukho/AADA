import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

torch.manual_seed(42)

class iMNIST(datasets.MNIST):
    def get_data(self):
        data = self.data.unsqueeze(1)
        data = data.expand(data.shape[0],3,data.shape[2],data.shape[3])
        return data
        
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(iMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
 
    def __getitem__(self, index):
        d, l = super(iMNIST, self).__getitem__(index)
        return d, l, index 



class iSVHN(datasets.SVHN):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        my_split = "train" if train==True else "test"
        super(iSVHN, self).__init__(root, split=my_split, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index):
        d, l = super(iSVHN, self).__getitem__(index)
        return d, l, index

svhn_transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
