import numpy as np
import pandas as pd
import matplotlib as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class MyDataSet(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = pd.read_csv(self.data_dir)

    def __getitem__(self, idx):
        X = self.data.iloc[idx, 0:13].values
        y = self.data.iloc[idx, 13]
        return torch.Tensor(X), torch.tensor([y]).float()

    def __len__(self):
        return self.data.shape[0]


my_data_set = MyDataSet("../Dataset-Mechine_Learning/heart.csv")

loader = DataLoader(my_data_set, batch_size=10)


def normalize(batch):
    batch_X = batch[0]
    batch_X = batch_X.float()  # 将数据类型转换为浮点型
    mean_X = torch.mean(batch_X)
    std_X = torch.std(batch_X)
    normalized_X = (batch_X - mean_X) / std_X
    return normalized_X


for batch in loader:
    batch_X = normalize(batch)
    batch_y = batch[1]
