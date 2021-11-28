"""Preprocessing script to apply onto all (image, label) pairs in the Cityscapes Dataset."""

import glob

import torch
import torchvision
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode, Resize

from spectr.src.utils import get_classId_from_rgb

trainImageFilePaths = glob.glob("/srv/datasets/cityscapes/leftImg8bit/train/**/*_leftImg8bit.png", recursive=True)
trainLabelFilePaths = glob.glob("/srv/datasets/cityscapes/gtFine/train/**/*_color.png", recursive=True)
valImageFilePaths = glob.glob("/srv/datasets/cityscapes/leftImg8bit/val/**/*_leftImg8bit.png", recursive=True)
valLabelFilePaths = glob.glob("/srv/datasets/cityscapes/gtFine/val/**/*_color.png", recursive=True)
testImageFilePaths = glob.glob("/srv/datasets/cityscapes/leftImg8bit/test/**/*_leftImg8bit.png", recursive=True)
testLabelFilePaths = glob.glob("/srv/datasets/cityscapes/gtFine/test/**/*_color.png", recursive=True)

resize = Resize((256, 512), interpolation=InterpolationMode.NEAREST)

# Loop through all image file paths. Modify and save in new dir.
# for filePath in trainImageFilePaths:
#     image = resize(read_image(filePath))
#     image = image / 255.0
#     fileName = filePath.split("/")[-1].replace(".png", "")
#     torch.save(image, f"/coc/scratch/aahluwalia30/cityscapes-processed/leftImg8bit/train/{fileName}.pt")
# print("Completed: Training Images Preprocessed.")

for filePath in valImageFilePaths:
    image = resize(read_image(filePath))
    image = image / 255.0
    fileName = filePath.split("/")[-1].replace(".png", "")
    torch.save(image, f"/coc/scratch/aahluwalia30/cityscapes-processed/leftImg8bit/val/{fileName}.pt")
print("Completed: Validation Images Preprocessed.")


# for filePath in testImageFilePaths:
#     image = resize(read_image(filePath))
#     image = image / 255.0
#     fileName = filePath.split("/")[-1].replace(".png", "")
#     torch.save(image, f"/coc/scratch/aahluwalia30/cityscapes-processed/leftImg8bit/test/{fileName}.pt")
# print("Completed: Testing Images Preprocessed.")


# Loop through all label file paths. Modify and save in new dir.
# for filePath in trainLabelFilePaths:
#     label = resize(read_image(filePath))
#     label = label[:-1, :, :]

#     # Generate 2d tensor of classIds from 3d semantically segmented label image.
#     label_classIds = torch.zeros(label.shape[1], label.shape[2])
#     for row in range(label.shape[1]):
#         for col in range(label.shape[2]):
#             label_classIds[row, col] = get_classId_from_rgb(label[:, row, col])

#     fileName = filePath.split("/")[-1].replace(".png", "")
#     torch.save(label_classIds, f"/coc/scratch/aahluwalia30/cityscapes-processed/gtFine/train/{fileName}.pt")
# print("Completed: Training Labels Preprocessed.")


for filePath in valLabelFilePaths:
    label = resize(read_image(filePath))
    label = label[:-1, :, :]

    # Generate 2d tensor of classIds from 3d semantically segmented label image.
    label_classIds = torch.zeros(label.shape[1], label.shape[2])
    for row in range(label.shape[1]):
        for col in range(label.shape[2]):
            label_classIds[row, col] = get_classId_from_rgb(label[:, row, col])

    fileName = filePath.split("/")[-1].replace(".png", "")
    torch.save(label_classIds, f"/coc/scratch/aahluwalia30/cityscapes-processed/gtFine/val/{fileName}.pt")
print("Completed: Validation Labels Preprocessed.")


# for filePath in testLabelFilePaths:
#     label = resize(read_image(filePath))
#     label = label[:-1, :, :]

#     # Generate 2d tensor of classIds from 3d semantically segmented label image.
#     label_classIds = torch.zeros(label.shape[1], label.shape[2])
#     for row in range(label.shape[1]):
#         for col in range(label.shape[2]):
#             label_classIds[row, col] = get_classId_from_rgb(label[:, row, col])

#     fileName = filePath.split("/")[-1].replace(".png", "")
#     torch.save(label_classIds, f"/coc/scratch/aahluwalia30/cityscapes-processed/gtFine/test/{fileName}.pt")
# print("Completed: Testing Labels Preprocessed.")
