"""Aggregates all of the training data in a2d2 directory and saves to disk."""

import os
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = "./../a2d2_semantic_dataset_subset"

print('Scanning for all images in A2D2 dataset directory...')
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
    cameraFilePath = imagePaths[idx+1]

    modifiedLabelPath = labelFilePath.replace("label", "")
    modifiedCameraPath = cameraFilePath.replace("camera", "")
    assert modifiedLabelPath == modifiedCameraPath

# Aggregate all features and labels into 2 separate lists.
road_scene_images = []
road_scene_labels = []

numPNGFiles = len(imagePaths)
numDataPairs = int(len(imagePaths) / 2)

print(f'Found {numPNGFiles} .png files total.')
print(f'Found {numDataPairs} image pairs in dataset.')

for idx in range(0, len(imagePaths), 2):
    labelFilePath = imagePaths[idx]
    cameraFilePath = imagePaths[idx+1]

    roadImage = cv2.imread(cameraFilePath)
    roadImage = cv2.resize(roadImage, (480, 304), interpolation=cv2.INTER_AREA)
    roadImage = cv2.cvtColor(roadImage, cv2.COLOR_BGR2RGB)
    roadImage = np.transpose(roadImage, (2, 0, 1))
    road_scene_images.append(roadImage)

    roadLabel = cv2.imread(labelFilePath)
    roadLabel = cv2.resize(roadLabel, (480, 304), interpolation=cv2.INTER_AREA)
    roadLabel = cv2.cvtColor(roadLabel, cv2.COLOR_BGR2RGB)
    roadLabel = np.transpose(roadLabel, (2, 0, 1))
    road_scene_labels.append(roadLabel)

    if (idx % 100 == 0):
        print(f'Loaded {idx+1}/{numDataPairs} images.')

road_scene_images = np.array(road_scene_images)
road_scene_labels = np.array(road_scene_labels)

print(f'Tensor shape of road scene images: {road_scene_images.shape}')
print(f'Tensor shape of road scene labels: {road_scene_labels.shape}')

np.save('./../a2d2_semantic_dataset_subset/data/road_scene_images', road_scene_images)
np.save('./../a2d2_semantic_dataset_subset/data/road_scene_labels', road_scene_labels)

print("Finished aggregating all A2D2 Road Scene Data.  :)")