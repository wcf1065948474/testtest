import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import h5py

class CelebaDataset_h5py(object):
    def __init__(self,opt):
        self.opt = opt
        self.wh = int(opt.full_size/opt.micro_size)-1
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))]
        )
        self.file = h5py.File(self.opt.datapath,'r',swmr=True)
        self.file_data = self.file['celeba']
    def __getitem__(self,index):
        self.macro_patches = []
        img = self.file_data[index % self.opt.max_dataset,:,:,:]
        img = self.transform(img)
        for i in range(self.wh):
            i *= self.opt.micro_size
            for j in range(self.wh):    
                j *= self.opt.micro_size
                patch = img[:,i:i+self.opt.macro_size,j:j+self.opt.macro_size]
                self.macro_patches.append(patch)
        return self.macro_patches
    def __len__(self):
        return self.opt.max_dataset
