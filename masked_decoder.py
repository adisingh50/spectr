import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import (
    MultiheadAttention,
    Linear,
    Dropout,
    LayerNorm,
    ReLU,
    Embedding,
    ModuleList,
)
import pdb


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        """
        Initializing necessary layers for a transformer layer
        """
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.self_attn = MultiheadAttention(d_model, nhead)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = ReLU()
        self.dropout_layer = Dropout(dropout)

    def forward(self, input, mask=None):
        """
        Defining forward pass for transformer layer
        Args:
            input: A tensor of shape [batch_size, num_patches, embed_dim]
            mask: A mask for self-attention
        Returns:
            output: A tensor of shape [batch_size, num_patches, embed_dim]
        """
        # pdb.set_trace()
        input_norm = self.norm1(input)
        if mask is not None:
            first_block = self.self_attn(input_norm, input_norm, input_norm, attn_mask=mask)[0] + input
        else:
            first_block = self.self_attn(input_norm, input_norm, input_norm)[0] + input
        norm_first_block = self.norm2(first_block)
        output = self.linear1(norm_first_block)
        output = self.activation(output)
        output = self.dropout_layer(output)
        output = self.linear2(output)
        output = self.dropout_layer(output)
        output += first_block
        return output


class MaskedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        patch_size: int = 8,
        num_layers: int = 6,
        num_channels: int = 64,
        num_classes: int = 38,
    ):
        super(MaskedTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.linear_embedder = Linear(patch_size * patch_size * num_channels, d_model)
        self.class_embedder = Embedding(num_classes, d_model)

        self.positional_embedder = Linear(1, d_model)

        self.transformer_layers = ModuleList(
            [TransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = LayerNorm(d_model)

    def forward(self, image_feature_maps):
        """
        Args:
            image_feature_maps: A tensor of size [batch_size, num_channels, height, width]
        Output:
            segmented_image: A tensor of size [batch_size, height, width]
        """
        pdb.set_trace()
        N, C, H, W = image_feature_maps.shape
        patches = image_feature_maps.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = torch.reshape(patches, [N, -1, self.patch_size * self.patch_size * self.num_channels])
        embedded_patches = self.linear_embedder(patches)
        x = torch.linspace(-1, 1, embedded_patches.shape[1])[:, None]
        positional_embeddings = self.positional_embedder(x)
        image_embeddings = embedded_patches + positional_embeddings
        class_embeddings = self.class_embedder(torch.arange(self.num_classes))[None]
        class_embeddings = class_embeddings.expand(image_embeddings.shape[0], -1, -1)
        combined_embeddings = torch.cat((image_embeddings, class_embeddings), dim=1)
        for decoder_block in self.transformer_layers:
            combined_embeddings = decoder_block(combined_embeddings)
        combined_embeddings = self.norm(combined_embeddings)

        processed_image_embeddings = combined_embeddings[:, : image_embeddings.shape[1], :]
        processed_class_embeddings = combined_embeddings[:, image_embeddings.shape[1] :, :]

        processed_image_embeddings /= processed_image_embeddings.norm(dim=-1, keepdim=True)
        processed_class_embeddings /= processed_class_embeddings.norm(dim=-1, keepdim=True)
        class_masks = torch.bmm(processed_image_embeddings, processed_class_embeddings.transpose(1, 2))
        class_masks = torch.reshape(
            class_masks,
            [N, H // self.patch_size, W // self.patch_size, self.num_classes],
        )

        class_masks = class_masks.permute(0, 3, 1, 2)
        upsampled_masks = F.interpolate(class_masks, scale_factor=[8, 8], mode="bilinear")
        normalized_masks = F.softmax(upsampled_masks, dim=3)

        return normalized_masks


if __name__ == "__main__":
    test_tensor = torch.randn(size=[4, 3, 480, 304])
    masked_transformer = MaskedTransformer(num_channels=test_tensor.shape[1])
    masked_transformer(test_tensor)
