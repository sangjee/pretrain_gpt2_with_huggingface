from torch.utils.data.dataset import Dataset
import torch

class HallymDataset(Dataset):

    def __init__(self, captions, tokenizer):
        self.captions = list(captions)
        self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=100)

    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
            }
        return item
