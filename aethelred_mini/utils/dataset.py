import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TokenDataset(Dataset):
    def __init__(self, data_path, seq_len=1024):
        # Data is assumed to be a numpy array of shape [N, seq_len]
        # Using mmap_mode='r' to avoid loading the entire dataset into RAM
        self.data = np.load(data_path, mmap_mode='r')
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We return input and target (shifted by 1)
        # Assuming the pre-chunked data has length seq_len
        row = self.data[idx]
        x = torch.from_numpy(row[:-1].astype(np.int64))
        y = torch.from_numpy(row[1:].astype(np.int64))
        return x, y

def get_dataloader(data_path, batch_size, seq_len, shuffle=True):
    dataset = TokenDataset(data_path, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
