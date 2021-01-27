# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torchvision.models as models
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
class Model(nn.Module):
    def __init__(self, pretrained_weights_path, device):
        super(Model, self).__init__()
        self.model = models.inception_v3()
        self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, 6)
        self.model.fc = nn.Linear(self.model.fc.in_features, 6)
        self.model.load_state_dict(
            torch.load(pretrained_weights_path, map_location=device))
        del self.model._modules['AuxLogits'] #删除AuxLogits模块
        #del self.model._modules['fc']
        #self.model.AuxLogits.fc = nn.Linear(self.model.AuxLogits.fc.in_features, self.args.n_classes) #将模型AuxLogits模块的fc输出通道数改成我们需要的分类数
        #print(self.model) #打印模型结构
        #print(self.model._modules.keys())  #可以打印出模型的所有模块名称
        #self.features = nn.Sequential(*list(self.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1))) #去掉最后一层fc层，这句也可以写成# del self.model._modules['fc']
        #self.features = nn.Sequential(*list(self.model.modules()))
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        self.last_node_num = 2048
            
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  #全局池化
        #self.classifier = nn.Linear(self.last_node_num, self.args.n_classes)  #最后加了一个全连接层
        
    def forward(self, x):  #重写forward函数，把几个模块组合起来
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        #x = self.classifier(x)
        return x


# %%
