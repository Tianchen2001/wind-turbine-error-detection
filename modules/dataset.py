import os
import numpy as np
import torch

from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, normal_path, whistle_path):
        normal_data = [os.path.join(normal_path, file_name) for file_name in os.listdir(normal_path)]
        whistle_data = [os.path.join(whistle_path, file_name) for file_name in os.listdir(whistle_path)]
        
        self.data = normal_data + whistle_data
        self.labels = np.concatenate([np.zeros(len(normal_data)), np.ones(len(whistle_data))])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = np.load(self.data[index])
        label = self.labels[index]
        
        return torch.from_numpy(data).float(), torch.tensor(label).long()