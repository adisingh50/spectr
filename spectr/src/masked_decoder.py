from typing import Optional
import pdb

import torch
import torch.nn as nn
from torch.nn import (
    MultiheadAttention,
    Linear,
    Dropout,
    LayerNorm,
    ReLU,
    Embedding,
    ModuleList,
)
import torch.nn.functional as F
import pytorch_lightning as pl


class TransformerLayer(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        """Initializing necessary layers for a transformer layer.

        Args:
            d_model: Embedding dimension
            nhead: Number of heads to be used for attention
            dim_feedforward: Dimension to be used for hidden linear layer
            dropout: Dropout rate
        """
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.self_attn = MultiheadAttention(d_model, nhead, batch_first = True)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = ReLU()
        self.dropout_layer = Dropout(dropout)

    def forward(
        self, input: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Defining forward pass for transformer layer

        Args:
            input: A tensor of shape [batch_size, num_patches, embed_dim]
            mask: A mask for self-attention
        Returns:
            output: A tensor of shape [batch_size, num_patches, embed_dim]
        """
        input_norm = self.norm1(input)
        if mask is not None:
            first_block = (
                self.self_attn(input_norm, input_norm, input_norm, attn_mask=mask)[0]
                + input
            )
        else:
            first_block = self.self_attn(input_norm, input_norm, input_norm)[0] + input
        norm_first_block = self.norm2(first_block)
        output = self.linear1(norm_first_block)
        output = self.activation(output)
        output = self.dropout_layer(output)
        output = self.linear2(output)
        output = self.dropout_layer(output)
        output = output + first_block
        return output


class MaskedTransformer(pl.LightningModule):
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
        """Initializes the Encoder object.

        Args:
            d_model: Embedding dimension
            nhead: number of heads to be used for multihead-attention
            dim_feedforward: Dimension to be used for hidden linear layer in each transformer block
            dropout: Dropout rate for each transformer block
            patch_size: Size of the patches used for splitting the image for attention
            num_layers: Number of transformer blocks
            num_channels: Number of channels in the input feature map
            num_classes: Number of distinct classes in the dataset
        """
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
            [
                TransformerLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model)

    def forward(self, image_feature_maps: torch.Tensor) -> torch.Tensor:
        """Feeds the input tensor forward through the Masked Transformer Decoder.

        Args:
            image_feature_maps: A tensor of size [batch_size, num_channels, height, width]
        Output:
            segmented_image: A tensor of size [batch_size, num_classes, height, width]
        """
        #pdb.set_trace()
        N, _, H, W = image_feature_maps.shape

        # Make sure height and width are fully divisible by patch size
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        # patches is now a 6D tensor with shape
        # [batch_size, num_channels, height // patch_size, width // patch_size, patch_size, patch_size]
        patches = image_feature_maps.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )

        # Reshapes patches into a 3D tensor of shape [batch_size, num_patches, flattened_patch_size]
        patches = torch.reshape(
            patches, [N, -1, self.patch_size * self.patch_size * self.num_channels]
        )
        embedded_patches = self.linear_embedder(patches)

        # This is another area where we differ from the research paper, they encode positional information differently
        # Adds positional information as the value of x at any given index is solely based on the value of that index
        x = torch.linspace(-1, 1, embedded_patches.shape[1], device = self.device)[:, None]
        positional_embeddings = self.positional_embedder(x)
        image_embeddings = embedded_patches + positional_embeddings

        class_embeddings = self.class_embedder(torch.arange(self.num_classes, device = self.device))[None]

        # Expands tensor of shape [num_classes, embed_dim] to [batch_size, num_classes, embed_dim]
        # Allows us to concatenate class embeddings with image embeddings
        # This is so the transformer can learn relations between different classes
        expanded_class_embeddings = class_embeddings.expand(image_embeddings.shape[0], -1, -1)
        combined_embeddings = torch.cat((image_embeddings, expanded_class_embeddings), dim=1)

        for decoder_block in self.transformer_layers:
            combined_embeddings = decoder_block(combined_embeddings)
        combined_embeddings = self.norm(combined_embeddings)

        processed_image_embeddings = combined_embeddings[
            :, : image_embeddings.shape[1], :
        ]
        processed_class_embeddings = combined_embeddings[
            :, image_embeddings.shape[1] :, :
        ]

        # Normalize image and class embeddings before multiplying
        processed_image_embeddings = processed_image_embeddings / processed_image_embeddings.norm(
            dim=-1, keepdim=True
        )
        processed_class_embeddings = processed_class_embeddings / processed_class_embeddings.norm(
            dim=-1, keepdim=True
        )

        class_masks = torch.bmm(
            processed_image_embeddings, processed_class_embeddings.transpose(1, 2)
        )

        # Initial Shape: [batch_size, num_patches, num_classes]
        # New Shape: [batch_size, height // patch_size, width // patch_size, num_classes]
        class_masks = torch.reshape(
            class_masks,
            [N, H // self.patch_size, W // self.patch_size, self.num_classes],
        )

        # Rearranging so channel is the 2nd dimension to make interpolation easier
        class_masks = class_masks.permute(0, 3, 1, 2)

        # Initial Shape: Upsamples from [batch_size, num_classes, height // patch_size, width // patch_size]
        # Upsampled Shape: [batch_size, num_classes, height, width]
        upsampled_masks = F.interpolate(
            class_masks, scale_factor=[8, 8], mode="bilinear"
        )
        #normalized_masks = F.softmax(upsampled_masks, dim=1)

        return upsampled_masks

class ContextTransformerLayer(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        """Initializing necessary layers for a transformer layer that uses cross attention in addition to self attention.

        Args:
            d_model: Embedding dimension
            nhead: Number of heads to be used for attention
            dim_feedforward: Dimension to be used for hidden linear layer
            dropout: Dropout rate
        """
        super(ContextTransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.self_attn = MultiheadAttention(d_model, nhead, dropout = dropout)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout = dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.activation = ReLU()
        self.dropout_layer = Dropout(dropout)

    def forward(
        self, input: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Defining forward pass for transformer layer

        Args:
            input: A tensor of shape [batch_size, num_patches, embed_dim]
            context: A tensor of shape [batch_size, num_channels, embed_dim]
            mask: A mask for self-attention
        Returns:
            output: A tensor of shape [batch_size, num_patches, embed_dim]
        """
        input = input.transpose(0, 1)
        input_norm = self.norm1(input)
        if mask is not None:
            first_block = (
                self.self_attn(input_norm, input_norm, input_norm, attn_mask=mask)[0]
                + input
            )
        else:
            first_block = self.self_attn(input_norm, input_norm, input_norm)[0] + input
        norm_first_block = self.norm2(first_block)
        if context is not None:
            context = context.transpose(0, 1)
            second_block = self.cross_attn(norm_first_block, context.float(), context.float())[0]
            norm_first_block = norm_first_block + second_block
            norm_first_block = self.norm3(norm_first_block)
        output = self.linear1(norm_first_block)
        output = self.activation(output)
        output = self.dropout_layer(output)
        output = self.linear2(output)
        output = self.dropout_layer(output)
        output = output + first_block
        return output.transpose(0, 1)

class ContextMaskedTransformer(pl.LightningModule):
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
        feature_map_height: int = 60,
        feature_map_width: int = 38,
    ):
        """Initializes the Encoder object.

        Args:
            d_model: Embedding dimension
            nhead: number of heads to be used for multihead-attention
            dim_feedforward: Dimension to be used for hidden linear layer in each transformer block
            dropout: Dropout rate for each transformer block
            patch_size: Size of the patches used for splitting the image for attention
            num_layers: Number of transformer blocks
            num_channels: Number of channels in the input feature map
            num_classes: Number of distinct classes in the dataset
        """
        super(ContextMaskedTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.linear_embedder = Linear(patch_size * patch_size * num_channels, d_model)
        self.feature_map_embedder = Linear(feature_map_height * feature_map_width, d_model)

        self.class_embedder = Embedding(num_classes, d_model)

        self.positional_embedder = Linear(1, d_model)

        self.transformer_layers = ModuleList(
            [
                ContextTransformerLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model)

    def forward(self, images: torch.Tensor, image_feature_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Feeds the input tensor forward through the Masked Transformer Decoder.

        Args:
            images: A tensor of size [batch_size, 3, height, width]
            image_feature_maps: A tensor of size [batch_size, num_channels, height // 8, width // 8]
        Output:
            segmented_image: A tensor of size [batch_size, num_classes, height, width]
        """
        N, _, H, W = images.shape

        # Make sure height and width are fully divisible by patch size
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        # patches is now a 6D tensor with shape
        # [batch_size, num_channels, height // patch_size, width // patch_size, patch_size, patch_size]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )

        # Reshapes patches into a 3D tensor of shape [batch_size, num_patches, flattened_patch_size]
        patches = torch.reshape(
            patches, [N, -1, self.patch_size * self.patch_size * self.num_channels]
        )
        embedded_patches = self.linear_embedder(patches)

        if image_feature_maps is not None:
            feature_map_flattened = torch.reshape(image_feature_maps, [image_feature_maps.shape[0], image_feature_maps.shape[1], -1])
            embedded_feature_maps = self.feature_map_embedder(feature_map_flattened)
        # This is another area where we differ from the research paper, they encode positional information differently
        # Adds positional information as the value of x at any given index is solely based on the value of that index
        x = torch.linspace(-1, 1, embedded_patches.shape[1], device = self.device)[:, None]
        positional_embeddings = self.positional_embedder(x)
        image_embeddings = embedded_patches + positional_embeddings

        class_embeddings = self.class_embedder(torch.arange(self.num_classes, device = self.device))[None]

        # Expands tensor of shape [num_classes, embed_dim] to [batch_size, num_classes, embed_dim]
        # Allows us to concatenate class embeddings with image embeddings
        # This is so the transformer can learn relations between different classes
        expanded_class_embeddings = class_embeddings.expand(image_embeddings.shape[0], -1, -1)
        combined_embeddings = torch.cat((image_embeddings, expanded_class_embeddings), dim=1)

        for decoder_block in self.transformer_layers:
            combined_embeddings = decoder_block(combined_embeddings, embedded_feature_maps)
        combined_embeddings = self.norm(combined_embeddings)

        processed_image_embeddings = combined_embeddings[
            :, : image_embeddings.shape[1], :
        ]
        processed_class_embeddings = combined_embeddings[
            :, image_embeddings.shape[1] :, :
        ]

        # Normalize image and class embeddings before multiplying
        processed_image_embeddings = processed_image_embeddings / processed_image_embeddings.norm(
            dim=-1, keepdim=True
        )
        processed_class_embeddings = processed_class_embeddings / processed_class_embeddings.norm(
            dim=-1, keepdim=True
        )

        class_masks = torch.bmm(
            processed_image_embeddings, processed_class_embeddings.transpose(1, 2)
        )

        # Initial Shape: [batch_size, num_patches, num_classes]
        # New Shape: [batch_size, height // patch_size, width // patch_size, num_classes]
        class_masks = torch.reshape(
            class_masks,
            [N, H // self.patch_size, W // self.patch_size, self.num_classes],
        )

        # Rearranging so channel is the 2nd dimension to make interpolation easier
        class_masks = class_masks.permute(0, 3, 1, 2)

        # Initial Shape: Upsamples from [batch_size, num_classes, height // patch_size, width // patch_size]
        # Upsampled Shape: [batch_size, num_classes, height, width]
        upsampled_masks = F.interpolate(
            class_masks, scale_factor=[self.patch_size, self.patch_size], mode="bilinear"
        )
        #normalized_masks = F.softmax(upsampled_masks, dim=1)

        return upsampled_masks
if __name__ == "__main__":
    test_tensor = torch.randn(size=[4, 3, 304, 480])
    test_feature_maps_tensor = torch.randn(size = [4, 128, 60, 38])
    masked_transformer = ContextMaskedTransformer(num_channels=test_tensor.shape[1], feature_map_height = test_feature_maps_tensor.shape[2], feature_map_width = test_feature_maps_tensor.shape[3])
    masked_transformer(test_tensor, test_feature_maps_tensor)
