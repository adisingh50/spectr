import torch
import torch.nn as nn
import numpy as np
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm, ReLU, Embedding

class TransformerLayer(nn.Module)
    def __init__(self, d_model: int = 256, nhead: int = 3, dim_feedforward: int = 512, dropout: float = 0.2):
        """
        Initializing necessary layers for a transformer layer
        """
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

    def forward(self, input, mask = None):
        """
        Defining forward pass for transformer layer
        Args:
            input: A tensor of shape [batch_size, num_patches, embed_dim]
            mask: A mask for self-attention
        Returns:
            output: A tensor of shape [batch_size, num_patches, embed_dim]
        """
        first_block = self.self_attn(self.norm1(input), mask) + input
        norm_first_block = self.norm2(first_block)
        output = self.linear1(norm_first_block)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.dropout(output)
        output += first_block
        return output

class MaskedTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 3, dim_feedforward: int = 512, dropout: float = 0.2, patch_size: int = 8, num_layers: int = 6, num_channels: int = 64, num_classes: int = 38):
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

        self.positional_embedder = nn.Linear(2, d_model)

        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
    
    def forward(self, image_feature_maps):
        """
        Args:
            image_feature_maps: A tensor of size [batch_size, num_channels, height, width]
        Output:
            segmented_image: A tensor of size [batch_size, height, width]
        """
        N, H, W, C = image_feature_maps.shape
        patches = image_feature_maps.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = torch.reshape(patches, [N, -1, self.patch_size * self.patch_size * self.num_channels])

        embedded_patches = self.linear_embedder(patches)
        
        class_embeddings = self.class_embedder(torch.arange(self.num_classes))


