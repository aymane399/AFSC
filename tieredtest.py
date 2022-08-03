import numpy as np
import torch
import os
from torch import nn
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from torch.utils.data import Dataset


np.random.seed(0)

L = []
arr = []
dataset_path = ".\datasets\data\\tiered_imagenet\data\\"

for file in os.listdir(dataset_path):
     filename = os.fsdecode(file)
     if filename.startswith("test"):
        if filename[8] == '_':
            f = filename[:8]
        if filename[9] == '_':
            f = filename[:9]
        elif filename[10] == '_':
            f = filename[:10]
        elif filename[11] == '_':
            f = filename[:11]
            
        if f not in L:
            print(f)
            arr.append(int(f[7:]))
            L.append(f)

classes = np.array([arr])
print(np.sort(classes))