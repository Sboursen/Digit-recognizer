# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:46:43 2020

@author: sboursen
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os




def save_model(model, num_epochs = 20, basemodel = "Efficientnet_B0_gridmask"):
    newpath = f'saved_models' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    PATH = newpath + f"/basicmodel_{basemodel}_statedict_{num_epochs}.pth"
    torch.save(model.state_dict(), f = PATH)

save_model(model=base_model)
    