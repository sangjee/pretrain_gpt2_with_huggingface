from torch.utils.data import DataLoader
import pytorch_lightning as pl
from HallymDataset import HallymDataset

class GPT2PretrainDataModule(pl.LightningDataModule):
  def __init__(self, train_df, val_df, test_df, batch_size, tokenizer, num_workers):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.num_workers = num_workers

  def setup(self, stage=None):
    self.train_dataset = build_loaders(self.train_df, self.tokenizer)
    self.val_dataset = build_loaders(self.val_df, self.tokenizer)
    self.test_dataset = build_loaders(self.test_df, self.tokenizer)

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=self.num_workers
    )

def build_loaders(dataframe, tokenizer):
    dataset = HallymDataset(
        dataframe['caption'].values,
        tokenizer=tokenizer
    )
    return dataset