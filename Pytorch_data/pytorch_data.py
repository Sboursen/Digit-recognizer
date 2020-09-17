# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:54:06 2020

@author: sboursen
"""

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch.nn.functional as F
import torch




class digitdataset(Dataset):
    ''' Implementing a pytorch dataset for the digit recognizer '''
    def __init__(self, data, dim = (28 , 28), transform = None ):
        ''' args:
            data : pandas dataframe (label + unrolled image)
            dim : image 2D shape default: (28 , 28)
            transform: bool, applying a transform
            
            '''
        self._dim = dim
        self._transform = transform
        self._data = data
        
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        image = self._data.iloc[index , 1:].values.reshape((*self._dim)).astype(np.float)
        image = Image.fromarray(image).convert("RGB")
        label = self._data.iloc[index, 0]
        
        if self._transform is not None:
            image = self._transform(image = np.array(image))
        
        label = F.one_hot(torch.tensor([label]), num_classes=10).squeeze()
            
        return image , label
            


