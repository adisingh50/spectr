"""File that loads in a Pytorch Model checkpoint and evaluates it on either a train, validation, or testing batch
from the Cityscapes dataset."""

import pdb
import os

import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
import torch
import torchmetrics
import torchvision

from spectr.src.utils import *

def evaluate_train_batch(model: pl.LightningModule):
    """Inferences the model on a batch of training data. Reports mIoU and generates visualizations for all images in 
    the training batch.

    Args:
        model (pl.LightningModule): a Pytorch Lightning Model.
    """
    print("Training Batch Results:")
    # Inference on Training Batch
    trainX = torch.load("./batches/x.pth").cpu()
    trainY = torch.load("./batches/y.pth").cpu()

    train_y_pred = model.forward(trainX)
    IOU = torchmetrics.IoU(num_classes=30, reduction="elementwise_mean")
    mIoU = IOU(train_y_pred, trainY.to(torch.int32))
    print(f"Model mIoU: {mIoU}")
    
    # Save Training Batch Visualizations
    os.makedirs("viz", exist_ok=True)
    os.makedirs("viz/images", exist_ok=True)
    os.makedirs("viz/ground_truth", exist_ok=True)
    os.makedirs("viz/pred_labels", exist_ok=True)

    print("Saving visualizations for all images in batch...")
    batch_size = trainX.shape[0]
    for i in range(batch_size):
        print(f'Visualizing Image {i}...')
        image = trainX[i, :, :, :]
        torchvision.utils.save_image(image, f"viz/images/image{i}.png")

        y_gt_2d = trainY[i, :, :]
        y_gt_3d = get_colored_label(y_gt_2d) / 255.0
        torchvision.utils.save_image(y_gt_3d, f"viz/ground_truth/gt{i}.png")

        y_pred = train_y_pred[i, :, :, :]
        y_pred_2d = torch.argmax(y_pred, dim=0)
        y_pred_3d = get_colored_label(y_pred_2d) / 255.0
        torchvision.utils.save_image(y_pred_3d, f"viz/pred_labels/spectr_pred{i}.png")
    
    print('Finished Visualizing Training Batch.')

def evaluate_val_batch(encoder_model: pl.LightningModule):
    """Inferences the model on a batch of validation data. Reports mIoU and generates visualizations for all images in 
    the validation batch.

    Args:
        model (pl.LightningModule): a Pytorch Lightning Model.
    """
    print("Validation Batch Results:")
    # Get a batch from validation set
    valX = torch.load("valX.pt")
    valY = torch.load("valY.pt")

    # Report mIoU
    val_y_pred = encoder_model.forward(valX)
    IOU = torchmetrics.IoU(num_classes=30, reduction="elementwise_mean")
    mIoU = IOU(val_y_pred, valY.to(torch.int32))
    print(f"Model mIoU: {mIoU}")
 

    # Save Validation Batch Visualizations
    os.makedirs("viz_val", exist_ok=True)
    os.makedirs("viz_val/images", exist_ok=True)
    os.makedirs("viz_val/ground_truth", exist_ok=True)
    os.makedirs("viz_val/pred_labels", exist_ok=True)

    print("Saving visualizations for all images in validation batch...")
    batch_size = valX.shape[0]
    for i in range(batch_size):
        print(f'Visualizing Image {i}...')
        image = valX[i, :, :, :]
        torchvision.utils.save_image(image, f"viz_val/images/image{i}.png")

        y_gt_2d = valY[i, :, :]
        y_gt_3d = get_colored_label(y_gt_2d) / 255.0
        torchvision.utils.save_image(y_gt_3d, f"viz_val/ground_truth/gt{i}.png")

        y_pred = val_y_pred[i, :, :, :]
        y_pred_2d = torch.argmax(y_pred, dim=0)
        y_pred_3d = get_colored_label(y_pred_2d) / 255.0
        torchvision.utils.save_image(y_pred_3d, f"viz_val/pred_labels/spectr_pred{i}.png") 
    print('Finished Visualizing Validation Batch.')

def evaluate_test_batch(encoder_model: pl.LightningModule):
    """Inferences the model on a batch of testing data. Reports mIoU and generates visualizations for all images in 
    the testing batch.

    Args:
        model (pl.LightningModule): a Pytorch Lightning Model.
    """
    print("Test Batch Results:")

    # Get the test batch
    testX = torch.load("testX.pt")
    testY = torch.load("testY.pt")

    pdb.set_trace()
    # Report mIoU
    test_y_pred = encoder_model.forward(testX)
    IOU = torchmetrics.IoU(num_classes=30, reduction="elementwise_mean")
    mIoU = IOU(test_y_pred, testY.to(torch.int32))
    print(f"Model mIoU: {mIoU}")
 
    # Save Validation Batch Visualizations
    os.makedirs("viz_test", exist_ok=True)
    os.makedirs("viz_test/images", exist_ok=True)
    os.makedirs("viz_test/ground_truth", exist_ok=True)
    os.makedirs("viz_test/pred_labels", exist_ok=True)

    print("Saving visualizations for all images in testing batch...")
    batch_size = testX.shape[0]
    for i in range(batch_size):
        print(f'Visualizing Image {i}...')
        image = testX[i, :, :, :]
        torchvision.utils.save_image(image, f"viz_test/images/image{i}.png")

        y_gt_2d = testY[i, :, :]
        y_gt_3d = get_colored_label(y_gt_2d) / 255.0
        torchvision.utils.save_image(y_gt_3d, f"viz_test/ground_truth/gt{i}.png")

        y_pred = test_y_pred[i, :, :, :]
        y_pred_2d = torch.argmax(y_pred, dim=0)
        y_pred_3d = get_colored_label(y_pred_2d) / 255.0
        torchvision.utils.save_image(y_pred_3d, f"viz_test/pred_labels/spectr_pred{i}.png") 
    print('Finished Visualizing Testing Batch.')


def main():
    # Init Data Module and Encoder Model
    torch.autograd.set_detect_anomaly(True)
    with hydra.initialize_config_module(config_module="spectr.config"):
        cfg = hydra.compose(config_name="config.yaml")
        spectr_config = instantiate(cfg.SpectrConfig)
    
    spectr_data_module = spectr_config.data_module
    encoder_model = spectr_config.custom_segmenter

    # Load Checkpoint
    checkpoint = torch.load("/srv/share2/aahluwalia30/spectr/lightning_logs/version_162183/checkpoints/epoch=44-step=16739.ckpt")
    encoder_model.load_state_dict(checkpoint["state_dict"])

    print('Model Loaded')
    # evaluate_train_batch(encoder_model)
    evaluate_val_batch(encoder_model)

if __name__ == "__main__":
    main()