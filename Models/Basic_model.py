# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:55:21 2020

@author: Asus
"""
# Imports


import torch
import torch.nn as nn
import torchvision.models as pretrainedmodels
from efficientnet_pytorch import EfficientNet





class Basicmodel(nn.Module):
    def __init__(self, pretrained_model = EfficientNet.from_name('efficientnet-b0'), num_classes = 10, pretrained = False):
        super(Basicmodel, self).__init__()
        
        
        # self.pretrainedmodel =  nn.Sequential(*list(pretrained_model.children())[:-4])
        
        self.pretrained_model = pretrained_model
        
        # for param in self.pretrained_model.parameters():
        #     param.requires_grad = True
        self.GAP = nn.AdaptiveAvgPool2d((1 , 1))
        
        self.bn = nn.BatchNorm2d(num_features = 1280, affine = True)
        self.relu = nn.ReLU(inplace = False)
        self.dropout= nn.Dropout(p = 0.2)
        self.linear = nn.Linear(in_features = 1280 , out_features = num_classes)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self , x):
        
        out = self.pretrained_model.extract_features(x)
        out = self.GAP(out).view((-1 , 1280))
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        
        out = self.softmax(out)
        
        
        return out



