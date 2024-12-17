# from torch.utils.data import Dataset, DataLoader
from jittor.dataset import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        super().__init__()
        self.data = data
        self.tkn = tokenizer
        self.mx_len = max_len
        self.total_len = len(data)
    
    def __getitem__(self, index):
        return self.data[index]
    


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        super().__init__()
        self.data = data
        self.tkn = tokenizer
        self.mx_len = max_len
        self.total_len = len(data)

    def __getitem__(self, index):
        return self.data[index]