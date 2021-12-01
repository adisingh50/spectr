"""Testing File that Overfits SPECTR model to 1 batch of data. Used to gain insight into learning performance."""

import pdb

import torch
import torch.nn as nn
import torchmetrics
import torchvision

from spectr.src.custom_segmenter import CustomSegmenter
from spectr.src.cityscapes_data_module import CityScapesDataModule
from spectr.src.utils import *


def main():
    # Hyperparameters
    encoder_config = {"k_width": 3, "pad": 1}
    decoder_config = {"d_model": 192, "nhead": 3, "num_channels": 64, "num_layers": 6, "num_classes": 30}
    batch_size = 8
    learning_rate = 0.01
    training_steps = 80000
    gamma = 1.0
    num_epochs = 1000

    # Define CityScapesDataModule
    train_dataset_path = "/coc/scratch/aahluwalia30/cityscapes_preprocessed/leftImg8bit/train/**/*_leftImg8bit.pt"
    train_labels_path = "/coc/scratch/aahluwalia30/cityscapes_preprocessed/gtFine/train/**/*_color.pt"
    val_dataset_path = "/coc/scratch/aahluwalia30/cityscapes_preprocessed/leftImg8bit/val/**/*_leftImg8bit.pt"
    val_labels_path = "/coc/scratch/aahluwalia30/cityscapes_preprocessed/gtFine/val/**/*_color.pt"
    test_dataset_path = "/coc/scratch/aahluwalia30/cityscapes_preprocessed/leftImg8bit/test/**/*_leftImg8bit.pt"
    test_labels_path = "/coc/scratch/aahluwalia30/cityscapes_preprocessed/gtFine/test/**/*_color.pt"
    batch_size = 8

    dataModule = CityScapesDataModule(
        train_dataset_path,
        train_labels_path,
        val_dataset_path,
        val_labels_path,
        test_dataset_path,
        test_labels_path,
        batch_size,
    )
    trainDataLoader = dataModule.train_dataloader()
    x, y = next(iter(trainDataLoader))
    x = x.cuda()
    y = y.cuda()

    torch.save(x, "x.pth")
    torch.save(y, "y.pth")

    # Define 3 main components
    model = CustomSegmenter(encoder_config, decoder_config, learning_rate, training_steps, gamma).cuda()
    print(get_num_parameters(model))

    class_weights = get_class_weights(y)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Main Training Loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = ce_loss(y_pred, y.to(torch.int64))
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}  Loss: {loss}")

    torch.save(model.state_dict(), "./overfit_model.pth")

    # Visualize images and predicted labels in batch
    print("Finished Overfit Training. Visualizing output.")
    model.eval()
    for i in range(8):
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
