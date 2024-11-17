import torch.nn as nn
from .rwkv_layer import RWKVLayer


class RWKV(nn.Module):

    def __init__(self, d_model, depth, d_ffn, num_classes):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.d_ffn = d_ffn
        self.num_classes = num_classes

        # backbone
        self.layers = nn.ModuleList([RWKVLayer(d_model=d_model, layer_id=i, num_layers=depth, dim_feedforward=d_ffn) for i in range(depth)])
        # head
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, input):
        # input.shape = [batch_size, ctx_len, d_model]
        # output.shape = [batch_size, num_classes]

        x = input
        # backbone
        for layer in self.layers:
            x = layer(x)
        # head
        x = x[:, -1, :]         # [batch_size, d_model]
        output = self.head(x)   # [batch_size, num_classes]
        
        return output