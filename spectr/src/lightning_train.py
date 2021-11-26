import pdb

import torch
import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate

from spectr.src.custom_segmenter import CustomSegmenter
from spectr.src.cityscapes_data_module import CityScapesDataModule
from spectr.spectr_config import SpectrConfig


def main():
    #pdb.set_trace()
    with hydra.initialize_config_module(config_module="spectr.config"):
        cfg = hydra.compose(config_name="initial_config.yaml")
        spectr_config = instantiate(cfg.SpectrConfig)
    
    spectr_data_module = spectr_config.data_module
    segmenter_model = spectr_config.custom_segmenter
    training_steps = spectr_config.training_steps

    batch_size = spectr_config.batch_size
    dataset_length = len(spectr_data_module.cityscapes_dataset_train)

    num_epochs = training_steps * batch_size // dataset_length
    trainer = pl.Trainer(accelerator = spectr_config.accelerator, gpus = spectr_config.num_gpus, max_epochs = num_epochs, fast_dev_run = True)
    trainer.fit(segmenter_model, spectr_data_module)

if __name__ == "__main__":
    main()