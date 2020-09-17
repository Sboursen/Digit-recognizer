# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:18:59 2020

@author: sboursen
"""


# =============================================================================
# Import localy defined function
# =============================================================================

import sys

sys.path.insert(1, 'D:/AIprojects/Digit-rocognizer/Pytorch_data/pytorch_data.py')
sys.path.insert(2, 'D:/AIprojects/Digit-rocognizer/data')

import albumentations
from Models.Basic_model import Basicmodel
from Pytorch_data.pytorch_data import digitdataset
from Pytorch_data.Augmentations import GridMask , RandomAugMix
from albumentations import (Compose, ToFloat)
from albumentations.augmentations.transforms import  ShiftScaleRotate, Resize
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader , Dataset
from Pytorch_data import pytorch_data
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm as tq
from torch.optim.lr_scheduler import StepLR






# seed everything
seed = 152
def seed_everything(seed = seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f'setting everything to seed {seed}')
    
seed_everything(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load and process the data

img_size = (56 , 56)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_df , val_df = train_test_split(train, test_size = 0.1,
                                      random_state = seed,
                                      stratify = train.iloc[: , 0])

train_df = train_df.reset_index(drop = True)
val_df = val_df.reset_index(drop = True)



train_augmentations = Compose([
    albumentations.OneOf([
        GridMask(num_grid=3, mode=0, rotate=15),
        GridMask(num_grid=3, mode=2, rotate=15),
    ], p=0.7),
    RandomAugMix(severity=4, width=3, alpha=1.0, p=0.7),
    Resize( *img_size),
    ShiftScaleRotate(shift_limit = 0, scale_limit = 0, rotate_limit = 10 , p = 0.5),
    ToFloat(max_value = 255),
    ToTensor()], p = 1)
val_augmentations = Compose([
    Resize(*img_size),
    ToFloat(max_value = 255),
    ToTensor()], p = 1)

train_dataset = digitdataset(data = train_df, transform = train_augmentations)
val_dataset = digitdataset(data = val_df , transform = val_augmentations)


# =============================================================================
# image , label = train_dataset.__getitem__(15)
# 
# plt.imshow(image["image"].permute(1 , 2 , 0).numpy(), cmap = 'gray')
# plt.title(str(label))
# =============================================================================





def trainmodel(base_model , num_epochs, batch_size = 64 ,
               train_dataset = train_dataset,
               val_dataset = val_dataset,
               num_workers = 0,
               l_rate = 0.001):
    """Training function."""
    train_loader = DataLoader(train_dataset , batch_size= batch_size,
                          num_workers = num_workers,
                          shuffle = True)
    val_loader = DataLoader(val_dataset , batch_size= batch_size,
                          num_workers = num_workers,
                          shuffle = True)

    model = base_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate)
    critarion = nn.BCELoss(reduction = 'mean')
    scheduler = StepLR(optimizer, step_size = 1, gamma = 0.75)
    
    train_loss , val_loss = [] , []
    train_acc , val_acc = [] , []
    
# =============================================================================
#     plt.ion()
#     
#     fig , axs = plt.subplots(nrows = 2 , sharex = True )
# =============================================================================
    

    
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}, ')
        model.train()
        running_loss = 0
        
        tq_train = tq(train_loader, total=int(len(train_loader)))
        y , preds = [] , []
        # x = np.arange(epoch + 1)
        
        
        for (images , labels) in tq_train:
            
            images = images["image"].to(device , dtype = torch.float)
            labels = labels.to(device , dtype = torch.float)
            optimizer.zero_grad()
            
            outputs = model(images).to(device , dtype = torch.float)
            

            loss = critarion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            y.extend(labels.cpu().detach().numpy().astype(int))
            preds.extend(outputs.cpu().detach().numpy())
            tq_train.set_postfix(loss=(loss.item()))
            
        epoch_loss = running_loss/(len(train_loader)/batch_size)
        train_loss.append(epoch_loss)
        preds = np.argmax(np.array(preds) , axis = 1).reshape(-1)
        preds  =np.eye(10)[preds]
        y = np.array(y)
        t_acc = (preds == y).mean()*100
        train_acc.extend([t_acc])
        
        
        
        # validation ------------------------------------------
        model.eval()
        running_loss = 0
        tq_val = tq(val_loader , total = int(len(val_loader)))
        
        
        y , preds = [] , []
        
        with torch.no_grad():
            for (images , labels) in tq_val:
            
                images = images["image"].to(device , dtype = torch.float)
                labels = labels.to(device , dtype = torch.float)
                outputs = model(images)
                loss = critarion(outputs, labels)
                running_loss += loss.item()
                y.extend(labels.cpu().numpy().astype(int))
                preds.extend(outputs.cpu().numpy())
                tq_val.set_postfix(loss=(loss.item()))
            
        val_epoch_loss = running_loss/(len(val_loader)/batch_size)
        val_loss.append(val_epoch_loss)
        
        preds = np.array(preds)
        preds = np.argmax(np.array(preds) , axis = 1).reshape(-1)
        preds  =np.eye(10)[preds]
        y = np.array(y)
        
        val_acc.extend([(preds == y).mean()*100])
        
        print(f"\n Epoch : {epoch}/{num_epochs} ,train_loss = {train_loss[epoch]:.2f}, val_loss = {val_loss[epoch]:.2f}, train_acc = {train_acc[epoch]:.2f}%, val_acc = {val_acc[epoch]:.2f}%")
        
        scheduler.step(val_epoch_loss)
        
    d = {'train_acc': train_acc, 'val_acc': val_acc }
    metrics = pd.DataFrame(d)
    metrics.to_csv(f"data/metrics.csv")
        
# =============================================================================
#         metrics_list = [train_loss , val_loss , train_acc , val_acc]
#         index = 0
#         for ax in axs:
#             ax.plot(x , metrics_list[index])
#             index += 1
# 
#         
#         fig.canvas.draw()
# =============================================================================
        
 
base_model = Basicmodel().to(device)
trainmodel(base_model = base_model , num_epochs = 20, batch_size = 256 )      
