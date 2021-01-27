# coding=utf-8                                                    
                                                                  
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image                                             
import numpy as np                                                
import torch                                                      
import os                                                         
                                                                  
class FundusDataset(Dataset):                                     
    def __init__(self, mode='train', root='', transform=None): 
        datapath = os.path.join(root, mode+'.txt')             
        with open(datapath) as fo:                                
            self.samples = fo.readlines()                         
            self.num_sample = len(self.samples)                   
        self.transform = transform                                
        self.y = []                                               
        for line in self.samples:                                 
            self.y.append(int(line.split('    ')[1]))             
                                                                  
    def __len__(self):                                            
        return self.num_sample                                    
                                                                  
    def __getitem__(self, idx):           
        sample = self.samples[idx].split('    ')                  
        imgpath = sample[0] 
        try:
            #img = Image.open(imgpath).convert('RGB')
            img = Image.open(imgpath)
        except Exception:
            print('bad img: ', imgpath)
        #img = transforms.ToTensor(img)
        label = int(sample[1])                                    
        if self.transform:                                        
            img = self.transform(img) 
        return img, label                                         
                                                                  
