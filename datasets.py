from torch.utils.data import Dataset
import random
import torch

class SupervisedWrapper(Dataset):
    def __init__(self, base_dataset, train_transform):
        self.base_dataset = base_dataset
        self.train_transform = train_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x_raw, y_raw = self.base_dataset[idx]
        x = self.train_transform(x_raw)
        return x, y_raw

    
    
class TwoViewWrapper(Dataset):
    def __init__(self, base_dataset, two_view_transform):
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x_raw, _ = self.base_dataset[idx]
        x1, x2 = self.two_view_transform(x_raw)
        return x1, x2


