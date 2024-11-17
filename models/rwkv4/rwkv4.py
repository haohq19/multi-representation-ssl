import torch.nn as nn
from .rwkv4_layer import RWKV4Layer


class RWKV4(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.depth = config['depth']
        self.d_ffn = config['d_ffn']
        self.num_classes = config['num_classes']

        # backbone
        self.layers = nn.ModuleList([RWKV4Layer(
            d_model=self.d_model,
            layer_id=i,
            num_layers=self.depth, 
            dim_feedforward=self.d_ffn) for i in range(self.depth)])
        # head
        self.head = nn.Linear(self.d_model, self.num_classes, bias=False)

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