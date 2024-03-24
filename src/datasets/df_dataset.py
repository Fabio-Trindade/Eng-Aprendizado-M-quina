
from torch.utils.data import Dataset
class DfDataset(Dataset):
    def __init__(self, df_features, df_labels, transform = None):
        self.x = df_features.values
        self.y = df_labels.values
        self.transform = transform
        self.len = len(self.x)

    def __getitem__(self,index):
        if self.transform:
            return self.transform(self.x[index]),self.transform(self.y[index])
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len
        
