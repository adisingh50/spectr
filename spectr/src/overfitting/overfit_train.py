"""Testing File that Overfits SPECTR model to 1 batch of data. Used to gain insight into learning performance."""

import pdb

import hydra
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torchmetrics
import torchvision

from spectr.src.custom_segmenter import CustomSegmenter
from spectr.src.cityscapes_data_module import CityScapesDataModule
from spectr.src.utils import *


def main():
    with hydra.initialize_config_module(config_module="spectr.config"):
        cfg = hydra.compose(config_name="initial_config.yaml")
        spectr_config = instantiate(cfg.SpectrConfig)

    # Define Model, Dataloader
    model = spectr_config.custom_segmenter.cuda()
    print(get_num_parameters(model))
    spectr_data_module = spectr_config.data_module
    trainDataLoader = spectr_data_module.train_dataloader()

    # Get a single batch
    x, y = next(iter(trainDataLoader))
    x = x.cuda()
    y = y.cuda()

    # Define Loss and Optimizer
    class_weights = get_class_weights(y)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    torch.save(x, "x.pth")
    torch.save(y, "y.pth")

    # Main Training Loop
    model.train()
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = ce_loss(y_pred, y.to(torch.int64))
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}  Loss: {loss}")

    # Save this model to file
    torch.save(model.state_dict(), "./overfit_model.pth")

    # Visualize images and predicted labels in batch
    print("Finished Overfit Training. Visualizing output.")
    model.eval()
    batch_size = x.shape[0]
    for i in range(batch_size):
        image = x[i, :, :, :]
        torchvision.utils.save_image(image, f"overfit_results/images/image{i}.png")

        y_gt_2d = y[i, :, :]
        y_gt_3d = get_colored_label(y_gt_2d) / 255.0
        torchvision.utils.save_image(y_gt_3d, f"overfit_results/ground_truth/gt{i}.png")

        image_input = image.unsqueeze(0)
        y_pred = model.forward(image_input)
        y_pred_2d = torch.argmax(y_pred, dim=1)[0]
        y_pred_3d = get_colored_label(y_pred_2d) / 255.0
        torchvision.utils.save_image(y_pred_3d, f"overfit_results/pred_labels/pred{i}.png")


if __name__ == "__main__":
    main()
