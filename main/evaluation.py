# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:46:09 2020

@author: sboursen
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader , Dataset
from Pytorch_data.pytorch_data import digitdataset
from albumentations import (Compose , ToFloat, Resize)
from albumentations.pytorch import ToTensor
from tqdm import tqdm as tq
import cv2

import torch.nn.functional as F
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = Basicmodel().to(device)



class test_digitdataset(Dataset):
    def __init__(self, data, dim = (28 , 28), transform = None):
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
        
        return image, label


def evaluate_model(model , statedict_PATH, num_images, img_size = (56 , 56)):
    
    """Evaluate model using some data """
    
    model.load_state_dict(torch.load(PATH))
    model.eval()

    batch_size = 256
    test_augmentations = Compose([
    Resize(*img_size),
    ToFloat(max_value = 255),
    ToTensor()], p = 1)
    
    test_df = pd.read_csv(f"data/test.csv")
    test_df = test_df.reset_index()
    test_dataset = test_digitdataset(data = test_df , transform = test_augmentations)
    test_loader = DataLoader(test_dataset , batch_size = batch_size , shuffle = False)
    
    test_tq = tq(test_loader , total = int(len(test_loader)))
    
    preds, labels = [], []
        
    with torch.no_grad():
        for (images , label) in test_tq:
            
            images = images["image"].to(device , dtype = torch.float)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy())
            labels.extend(label.cpu().numpy() + 1)
            
            
    
    preds = np.array(preds)
    preds = np.argmax(np.array(preds) , axis = 1).reshape(-1)

    
    
    fig , axes = plt.subplots(nrows = num_images//4 + 1 , ncols = 4,  figsize=(64,64), sharex = True, sharey = True)
    
    counter = 0
    for row in axes:
        for col in row:
            col.imshow(images[counter].squeeze().detach().permute(1 , 2 , 0).cpu().numpy())
            col.set_title(f"pred = {preds[counter]}")
            counter += 1
    
    test_preds = pd.read_csv(f"data/sample_submission.csv")
    test_preds.ImageId = labels
    test_preds.Label = preds
    
    save_file = f"data/sample_submission_temp.csv"
    if os.path.exists(save_file):
        
        os.remove(save_file)
    test_preds.to_csv(save_file , index= False)
    print("Submission file created successfully")
    
PATH = f"saved_models/basicmodel_Efficientnet_B0_gridmask_statedict_20.pth"
evaluate_model(model = model , statedict_PATH = PATH  , num_images = 16 , img_size= (56 , 56))