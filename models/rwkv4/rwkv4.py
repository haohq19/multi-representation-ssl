import torch
import torch.nn as nn
from .rwkv4_layer import RWKV4Layer


class RWKV4(nn.Module):

    def __init__(self, d_model: int, depth: int, d_ffn: int, num_classes: int):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.d_ffn = d_ffn
        self.num_classes = num_classes

        # backbone
        self.layers = nn.ModuleList([RWKV4Layer(
            d_model=self.d_model,
            layer_id=i,
            num_layers=self.depth, 
            dim_feedforward=self.d_ffn) for i in range(self.depth)])
        # head
        self.head = nn.Linear(self.d_model, self.num_classes, bias=False)

    def forward(self, input, return_hidden=False):
        # input.shape = [batch_size, ctx_len, d_model]
        # output.shape = [batch_size, num_classes]

        if return_hidden:
            hidden = []
            x = input
            for layer in self.layers:
                x, h = layer(x, return_hidden=True)
                hidden.append(h)
            x = x[:, -1, :]         # [batch_size, d_model]
            output = self.head(x)
            hidden = torch.cat(hidden, dim=1)  # [batch_size, depth * d_model]
            return output, hidden

        x = input
        # backbone
        for layer in self.layers:
            x = layer(x)
        # head
        x = x[:, -1, :]         # [batch_size, d_model]
        output = self.head(x)   # [batch_size, num_classes]
        
        return output