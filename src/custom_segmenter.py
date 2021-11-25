from typing import Dict, Any
import pdb

import torch
import torch.nn as nn
from encoder_network import EncoderNetwork
from masked_decoder import MaskedTransformer
from utils import get_num_parameters


class CustomSegmenter(nn.Module):
    def __init__(self, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any]):
        """Initializes the encoder-decoder CNN network and the decoder masked transformer network

        Args:
            encoder_config: All the necessary parameters to initialize the encoder
            decoder_config: All the necessary parameters to initialize the decoder
        """
        super(CustomSegmenter, self).__init__()

        self.encoder = EncoderNetwork(**encoder_config)
        self.decoder = MaskedTransformer(**decoder_config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Generates segmentation map for image

        Args:
            image: A tensor of shape [batch_size, num_channels, height, width]

        Returns:
            segmented_map: A tensor of shape [batch_size, num_classes, height, width]
        """

        image_feature_map = self.encoder(image)
        segmented_feature_map = self.decoder(image_feature_map)

        return segmented_feature_map


if __name__ == "__main__":
    pdb.set_trace()
    test_batch = torch.randn(size=[4, 3, 480, 304])
    encoder_config = {"k_width": 3, "pad": 1}
    decoder_config = {"num_channels": 64, "num_layers": 2}

    custom_segmenter = CustomSegmenter(encoder_config, decoder_config)
    total_parameters = get_num_parameters(custom_segmenter)
    print(f"Total number of parameters is {total_parameters}")
    output_segmentation = custom_segmenter(test_batch)
