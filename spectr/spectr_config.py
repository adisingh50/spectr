from typing import Dict, Any

import torch
import torch.nn as nn
from spectr.src.custom_segmenter import CustomSegmenter
from spectr.src.cityscapes_data_module import CityScapesDataModule

class SpectrConfig:
    def __init__(self, train_dataset_path: str, train_labels_path: str, val_dataset_path: str, val_labels_path: str, test_dataset_path: str, test_labels_path: str, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any], batch_size: int, learning_rate: float, training_steps: int, gamma: float, accelerator: str):
        self.accelerator = accelerator
        self.num_gpus = torch.cuda.device_count()
        self.training_steps = training_steps
        self.batch_size = batch_size

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        if self.accelerator.startswith("ddp"):
            self.data_module = CityScapesDataModule(train_dataset_path, train_labels_path, val_dataset_path, val_labels_path, test_dataset_path, test_labels_path, batch_size // self.num_gpus)
        else:
            self.data_module = CityScapesDataModule(train_dataset_path, train_labels_path, val_dataset_path, val_labels_path, test_dataset_path, test_labels_path, batch_size)
        
        self.custom_segmenter = CustomSegmenter(encoder_config, decoder_config, learning_rate, training_steps, gamma)
