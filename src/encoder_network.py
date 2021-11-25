"""Encoder-Decoder Class containing Convolutional Layers."""

import numpy as np
import torch
import torch.nn as nn
import torchvision


class EncoderDecoder(nn.Module):
    def __init__(self, k_width=3, pad=1) -> None:
        """Initializes the Encoder object.

        Args:
            k_width (int, optional): Height and width of convolutional kernel. Defaults to 3.
            pad (int, optional): Zero-padding applied to feature maps. Defaults to 1.
        """
        super(EncoderDecoder, self).__init__()

        self.relu = nn.ReLU()
        # Encoder: 5 Conv Layers
        self.encoder_conv1 = nn.Conv2d(
            3, 64, kernel_size=k_width, stride=1, padding=pad
        )
        self.encoder_bn1 = nn.BatchNorm2d(64)

        self.encoder_conv2 = nn.Conv2d(
            64, 128, kernel_size=k_width, stride=1, padding=pad
        )
        self.encoder_conv3 = nn.Conv2d(
            128, 128, kernel_size=k_width, stride=1, padding=pad
        )
        self.encoder_bn2 = nn.BatchNorm2d(128)

        self.encoder_conv4 = nn.Conv2d(
            128, 256, kernel_size=k_width, stride=1, padding=pad
        )
        self.encoder_conv5 = nn.Conv2d(
            256, 256, kernel_size=k_width, stride=1, padding=pad
        )
        self.encoder_bn3 = nn.BatchNorm2d(256)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder: 5 Transposed Conv Layers
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder_conv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=k_width, stride=1, padding=pad
        )
        self.decoder_conv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=k_width, stride=1, padding=pad
        )
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_bn2 = nn.BatchNorm2d(128)

        self.decoder_conv3 = nn.ConvTranspose2d(
            128, 128, kernel_size=k_width, stride=1, padding=pad
        )
        self.decoder_conv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=k_width, stride=1, padding=pad
        )
        self.decoder_bn3 = nn.BatchNorm2d(64)

        self.decoder_conv5 = nn.ConvTranspose2d(
            64, 64, kernel_size=k_width, stride=1, padding=pad
        )

    def forward(self, x: torch.Tensor):
        """Feeds the input tensor forward through the CNN Encoder and Decoder.

        Args:
            x (torch.Tensor): Batch of RGB Images, shape (N,C,H,W).
                N: batch size
                C: 3
                H: image height
                W: image width

        Returns:
            The convolved feature map as output, shape (N,H,W,C).
        """

        # Encoder feed forward
        x = self.relu(self.encoder_bn1(self.encoder_conv1(x)))
        x, max_indices_1 = self.maxpool(x)

        x = self.relu(self.encoder_bn2(self.encoder_conv2(x)))
        x = self.relu(self.encoder_bn2(self.encoder_conv3(x)))
        x, max_indices_2 = self.maxpool(x)

        x = self.relu(self.encoder_bn3(self.encoder_conv4(x)))
        x = self.relu(self.encoder_bn3(self.encoder_conv5(x)))

        # Decoder feed forward
        x = self.relu(self.decoder_bn1(self.decoder_conv1(x)))
        x = self.relu(self.decoder_bn2(self.decoder_conv2(x)))

        x = self.maxunpool(x, max_indices_2)
        x = self.relu(self.decoder_bn2(self.decoder_conv3(x)))
        x = self.relu(self.decoder_bn3(self.decoder_conv4(x)))

        x = self.maxunpool(x, max_indices_1)
        output = self.relu(self.decoder_bn3(self.decoder_conv5(x)))

        return output


# Just some dummy test code to test out shapes of I/O tensors. Will remove in the future.
if __name__ == "__main__":
    img1 = torchvision.io.read_image("sample1.png")
    img2 = torchvision.io.read_image("sample2.png")

    resize = torchvision.transforms.Resize((304, 480))
    img1 = resize(img1)
    img2 = resize(img2)

    img1 = torch.unsqueeze(img1, dim=0)
    img2 = torch.unsqueeze(img2, dim=0)
    batch_input = torch.cat((img1, img2), dim=0) / 255.0

    print(batch_input.shape)
    encoder = EncoderDecoder()
    output = encoder.forward(batch_input)
    print(output.shape)
