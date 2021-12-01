"""Testing File meant to evaluate performance of an overfit model."""

import pdb
import hydra
from hydra.utils import instantiate
import torch
import torchmetrics

from spectr.src.custom_segmenter import CustomSegmenter
from spectr.src.cityscapes_data_module import CityScapesDataModule
from spectr.spectr_config import SpectrConfig


def main():
    with hydra.initialize_config_module(config_module="spectr.config"):
        cfg = hydra.compose(config_name="initial_config.yaml")
        spectr_config = instantiate(cfg.SpectrConfig)

    spectr_data_module = spectr_config.data_module
    overfit_model = spectr_config.custom_segmenter
    overfit_model.load_state_dict(torch.load("overfit_model.pth"))

    # Feed forward on x
    x = torch.load("x.pth").cpu()
    y = torch.load("y.pth").cpu()

    y_pred = overfit_model.forward(x)

    # Calculate mIoU
    IOU = torchmetrics.IoU(num_classes=30)
    iou = IOU(y_pred, y.to(torch.int32))
    print(f"Overfit Model IoU: {iou}")


if __name__ == "__main__":
    main()
