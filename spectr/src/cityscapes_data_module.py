from typing import Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from spectr.src.cityscapes_dataset import CityScapesDataSet


class CityScapesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        train_labels_dir: str,
        vaL_data_dir: str,
        val_labels_dir: str,
        test_data_dir: str,
        test_labels_dir: str,
        batch_size: int,
    ):
        """Initializes DataModule for Cityscapes dataset that returns associated dataloaders

        Args:
            data_dir:
        """
        super().__init__()

        self.train_data_dir = train_data_dir
        self.train_labels_dir = train_labels_dir
        self.val_data_dir = vaL_data_dir
        self.val_labels_dir = val_labels_dir
        self.test_data_dir = test_data_dir
        self.test_labels_dir = test_labels_dir

        self.batch_size = batch_size

        self.cityscapes_dataset_train = CityScapesDataSet(train_data_dir, train_labels_dir)
        self.cityscapes_dataset_val = CityScapesDataSet(vaL_data_dir, val_labels_dir)
        self.cityscapes_dataset_test = CityScapesDataSet(test_data_dir, test_labels_dir)

    def setup(self, stage: Optional = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.cityscapes_dataset_train, self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.cityscapes_dataset_val, self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cityscapes_dataset_test, self.batch_size, shuffle=False, num_workers=2)
