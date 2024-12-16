import numpy as np
from torch.utils import data
import torch


class DataSet(data.Dataset):
    def __init__(self,data_dir, txt_path):
        self.x_list = []
        self.y_list = []
        with open(txt_path) as reader:
            lines = reader.readlines()
            for line in lines:
                data_name,label=line.strip('\n').split(',')
                x = '{}/{}.npy'.format(data_dir,data_name)
                y = int(float(label))

                self.x_list.append(x)
                self.y_list.append(y)

    def __len__(self):
        return len(self.y_list)

    def __getitem__(self, index):
        x = torch.from_numpy(np.load(self.x_list[index]))
        y = torch.tensor(self.y_list[index], dtype=torch.long)
        return x, y
