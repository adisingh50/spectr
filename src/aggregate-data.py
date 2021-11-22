"""Aggregates all of the training data in a2d2 directory and saves to disk. Saves road scene data 
and labels at spectr/a2d2_semantic_dataset_subset/data/."""

import os
from pathlib import Path

import numpy as np
import torch
import torchvision

DATA_ROOT = Path(__file__).parent.parent.resolve() / "a2d2_semantic_dataset_subset"

print("Scanning for all images in A2D2 dataset directory...")
# Collect all file paths under the a2d2 directory and then filter for only .png images (features + labels).
allFilePaths = []
for root, dirs, files in os.walk(DATA_ROOT, topdown=True):
    for fileName in files:
        allFilePaths.append(os.path.join(root, fileName))
imagePaths = [filePath for filePath in allFilePaths if "png" in filePath]


# Verify imagePaths has all (label, image) pairs of road scene images with proper file names.
# (label_1, camera_1, label_2, camera_2, ...., label_n, camera_n)
assert len(imagePaths) % 2 == 0

for idx in range(0, len(imagePaths), 2):
    labelFilePath = imagePaths[idx]
    cameraFilePath = imagePaths[idx + 1]

    modifiedLabelPath = labelFilePath.replace("label", "")
    modifiedCameraPath = cameraFilePath.replace("camera", "")
    assert modifiedLabelPath == modifiedCameraPath

# Aggregate all features and labels into 2 separate lists.
road_scene_images = []
road_scene_labels = []

numPNGFiles = len(imagePaths)
numDataPairs = int(len(imagePaths) / 2)

print(f"Found {numPNGFiles} .png files total.")
print(f"Found {numDataPairs} image pairs in dataset.")
resize = torchvision.transforms.Resize((304, 480))

for idx in range(0, len(imagePaths), 2):
    if idx % 100 == 0:
        print(f"Loaded {idx}/{numDataPairs} images.")

    labelFilePath = imagePaths[idx]
    cameraFilePath = imagePaths[idx + 1]

    roadImage = resize(torchvision.io.read_image(cameraFilePath))
    road_scene_images.append(roadImage)

    roadLabel = resize(torchvision.io.read_image(labelFilePath))
    road_scene_labels.append(roadLabel)

road_scene_images = torch.stack((road_scene_images))
road_scene_labels = torch.stack((road_scene_labels))

print(f"Tensor shape of road scene images: {road_scene_images.shape}")
print(f"Tensor shape of road scene labels: {road_scene_labels.shape}")

torch.save(road_scene_images, DATA_ROOT / "data/road_scene_images.pt")
torch.save(road_scene_labels, DATA_ROOT / "data/road_scene_labels.pt")

print("Finished aggregating all A2D2 Road Scene Data.  :)")
