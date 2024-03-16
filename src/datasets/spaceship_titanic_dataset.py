import numpy as np
import pandas as pd
from torch.utils.data import Dataset
class SpaceshipTitanicDataset(Dataset):
    def __init__(self, data, transform = None,type ='train'):
        if type == 'train':
            self.y = np.array(data,dtype=float)[:,-1]
            self.x = np.array(data,dtype=float)[:,:-1]
        else:
            self.x = np.array(data,dtype=float)
            self.y = np.array(data,dtype=float)[:,-1]
        self.transform = transform
        self.len = len(self.x)

    def __getitem__(self,index):
        if self.transform:
            return self.transform(self.x[index]),self.transform(self.y[index])
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len
        
