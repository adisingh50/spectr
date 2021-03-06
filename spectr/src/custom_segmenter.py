from typing import Dict, List, Any
import pdb

import torch
import torch.nn as nn
import pytorch_lightning as pl
from spectr.src.encoder_network import EncoderNetwork
from spectr.src.resnet_feature_extractor import SpectrResnet
from spectr.src.masked_decoder import ContextMaskedTransformer
from spectr.src.utils import get_num_parameters


class CustomSegmenter(pl.LightningModule):
    def __init__(
        self,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        learning_rate: float,
        training_steps: int,
        gamma: float,
    ):
        """Initializes the encoder-decoder CNN network and the decoder masked transformer network

        Args:
            encoder_config: All the necessary parameters to initialize the encoder
            decoder_config: All the necessary parameters to initialize the decoder
            learning_rate: Initial learning rate for optimizer
            training_steps: Number of batches for model to process during training
            gamma: Decay rate
        """
        super(CustomSegmenter, self).__init__()

        #self.encoder = EncoderNetwork(**encoder_config)
        #self.decoder = MaskedTransformer(**decoder_config)
        self.encoder = SpectrResnet()
        self.decoder = ContextMaskedTransformer(**decoder_config)
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Generates segmentation map for image

        Args:
            image: A tensor of shape [batch_size, num_channels, height, width]

        Returns:
            segmented_map: A tensor of shape [batch_size, num_classes, height, width]
        """

        image_feature_map = self.encoder(image.to(torch.float32))
        segmented_feature_map = self.decoder(image, image_feature_map)

        return segmented_feature_map

    def training_step(self, curr_batch: List[torch.Tensor], batch_idx: int) -> float:
        """Training step method called by Pytorch Lightning

        Args:
            curr_batch: A list of tensors (x, y). x is a tensor of shape [batch_size, num_channels, height, width], y is a tensor of shape [batch_size, height, width]

        Returns:
            output_loss: A number indicating cross entropy loss
        """
        #pdb.set_trace()
        x, y = curr_batch
        output = self(x)
        output_loss = self.loss(output, y.to(torch.int64))
        self.log("train_loss", output_loss)
        return output_loss

    def validation_step(self, val_batch: List[torch.Tensor], batch_idx: int) -> float:
        """Validation step method called by Pytorch Lightning

        Args:
            val_batch: curr_batch: A list of tensors (x, y). x is a tensor of shape [batch_size, num_channels, height, width], y is a tensor of shape [batch_size, height, width]

        Returns:
            output_loss: A number indicating cross entropy loss
        """

        x, y = val_batch
        with torch.no_grad():
            output = self(x)
            output_loss = self.loss(output, y.to(torch.int64))
        self.log("val_loss", output_loss)
        return output_loss

    def configure_optimizers(self):
        """Method called by Pytorch Lightning to set up optimizer and learning rate scheduler"""

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.training_steps // 20, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    pdb.set_trace()
    test_batch = torch.randn(size=[4, 3, 256, 512])
    encoder_config = {"k_width": 3, "pad": 1}
    decoder_config = {"num_channels": 3, "num_layers": 2}

    custom_segmenter = CustomSegmenter(encoder_config, decoder_config, learning_rate = 0.001, training_steps = 80000, gamma = 0.995)
    total_parameters = get_num_parameters(custom_segmenter)
    print(f"Total number of parameters is {total_parameters}")
    output_segmentation = custom_segmenter(test_batch)
