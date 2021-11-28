"""Class to load the Cityscapes dataset from within SkyNet file system."""

import glob
from typing import Tuple

import torch
from torch.utils.data import Dataset

from spectr.src.utils import get_classId_from_rgb


class CityScapesDataSet(Dataset):
    def __init__(self, imagesRoot: str, labelsRoot: str) -> None:
        """Initializes a CityScapesDataset instance.

        Args:
            imagesRoot (str): Root directory path for road scene images.
            labelsRoot (str): Root directory path for semantically segmented labels.
        """
        self.imageFilePaths = glob.glob(imagesRoot, recursive=True)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets image and label from the image paths list at the specificied index.

        Args:
            index (int): Index of (image, label) pair to retrieve.

        Raises:
            IndexError: if index is out of bounds of image paths list.

        Returns:
            image: road scene image tensor of shape (channels, height, width).
            label: tensor containing classIds for each pixel, shape (height, width).
        """
        if index >= len(self.imageFilePaths):
            raise IndexError("Index is out of bounds.")

        # String formatting on imageFilePath to convert it over to a semantic label file path.
        # Example:
        # /srv/datasets/cityscapes/leftImg8bit/train/bremen/bremen_000157_000019_leftImg8bit.png
        # /srv/datasets/cityscapes/gtFine/train/bremen/bremen_000157_000019_gtFine_color.png
        imageFilePath = self.imageFilePaths[index]
        expectedLabelFilePath = imageFilePath.replace("leftImg8bit", "gtFine", 1)
        expectedLabelFilePath = expectedLabelFilePath.replace("leftImg8bit", "gtFine_color", 1)

        image = torch.load(imageFilePath)
        label = torch.load(expectedLabelFilePath)

        return image, label

    def __len__(self) -> int:
        """Gets the length of the road scene images dataset.

        Returns:
            Length of image file paths list.
        """
        return len(self.imageFilePaths)


if __name__ == "__main__":
    imagesRoot = "/srv/datasets/cityscapes/leftImg8bit/train/**/*_leftImg8bit.png"
    labelsRoot = "/srv/datasets/cityscapes/gtFine/train/**/*_color.png"
    trainDataSet = CityScapesDataSet(imagesRoot, labelsRoot)
    image, label = trainDataSet.__getitem__(1020)

    print(trainDataSet.__len__())
    print(image.shape)
    print(label.shape)
    print(label[50:60, 70:80])
