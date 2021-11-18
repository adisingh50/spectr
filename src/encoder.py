"""Encoder Class containing Convolutional Layers. Based off VGG-16 Encoder."""

import cv2
import numpy as np
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, k_width=3, pad=1) -> None:
        """Initializes the Encoder object.

        Args:
            k_width (int, optional): Height and width of convolutional kernel. Defaults to 3.
            pad (int, optional): Zero-padding applied to feature maps. Defaults to 1.
        """
        super(Encoder, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=k_width, stride=1, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=k_width, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(64)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=k_width, stride=1, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=k_width, stride=1, padding=pad)
        self.bn2 = nn.BatchNorm2d(128)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=k_width, stride=1, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=k_width, stride=1, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=k_width, stride=1, padding=pad)
        self.bn3 = nn.BatchNorm2d(256)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=k_width, stride=1, padding=pad)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=k_width, stride=1, padding=pad)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=k_width, stride=1, padding=pad)
        self.bn4 = nn.BatchNorm2d(512)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=k_width, stride=1, padding=pad)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=k_width, stride=1, padding=pad)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=k_width, stride=1, padding=pad)
        self.bn5 = nn.BatchNorm2d(512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """Feeds the input tensor forward through the Encoder.

        Args:
            x (torch.Tensor): Batch of RGB Images, shape (N,C,H,W).
                N: batch size
                C: 3
                H: image height
                W: image width

        Returns:
            The convolved feature map as output.
        """
        x = self.relu(self.bn1(self.conv1_1(x)))
        x = self.relu(self.bn1(self.conv1_2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv2_1(x)))
        x = self.relu(self.bn2(self.conv2_2(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn3(self.conv3_1(x)))
        x = self.relu(self.bn3(self.conv3_2(x)))
        x = self.relu(self.bn3(self.conv3_3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn4(self.conv4_1(x)))
        x = self.relu(self.bn4(self.conv4_2(x)))
        x = self.relu(self.bn4(self.conv4_3(x)))
        x = self.maxpool(x)

        x = self.relu(self.bn5(self.conv5_1(x)))
        x = self.relu(self.bn5(self.conv5_2(x)))
        x = self.relu(self.bn5(self.conv5_3(x)))
        output = self.maxpool(x)

        return x

# Just some dummy test code to test out shapes of I/O tensors. Will remove in the future.
if __name__ == "__main__":
    img1 = cv2.imread('sample1.png')
    img2 = cv2.imread('sample2.png')


    img1 = cv2.resize(img1, (358, 225), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (358, 225), interpolation=cv2.INTER_AREA)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    batch = np.concatenate((img1, img2), axis=0)

    batch = torch.from_numpy(batch)
    batch_input = batch.permute(0, 3, 1, 2) / 255.0

    encoder = Encoder()
    output = encoder.forward(batch_input)
    print(output.shape)