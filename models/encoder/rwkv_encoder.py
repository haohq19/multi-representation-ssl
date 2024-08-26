import torch
import torch.nn as nn
import torch.nn.functional as F
from ..rwkv4.rwkv_layer import RWKVLayer



class EventNet(nn.Module):

    def __init__(self, patch_size, in_chans, d_embed):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_embed = d_embed
        self.vocab_size = in_chans * patch_size ** 2 + 1  # for events, in_chans = 2, the auxiliary one is for padding

        self.embedding = nn.Embedding(self.vocab_size, d_embed, padding_idx=0)

        self.ln = nn.LayerNorm(d_embed)

        self.fc = nn.Linear(d_embed, d_embed, bias=False)


    def forward(self, input):
        # input.shape = [batch_size, ctx_len, num_events]
        # output.shape = [batch_size, ctx_len, d_embed]

        x = self.embedding(input)   # [batch_size, ctx_len, num_events, d_embed]
        x = self.ln(x)              # [batch_size, ctx_len, num_events, d_embed]
        x, _ = torch.max(x, dim=2)  # [batch_size, ctx_len, d_embed]
        output = self.fc(x)         # [batch_size, ctx_len, d_embed]

        return output
        


class RWKV(nn.Module):

    def __init__(self, d_model, depth, d_ffn, num_classes):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.d_ffn = d_ffn
        self.num_classes = num_classes

        # rwkv backbone
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


class RWKVEncoder(nn.Module):

    def __init__(self, patch_size, in_chans, d_model, depth, d_ffn, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_model = d_model
        self.depth = depth
        self.d_ffn = d_ffn
        self.num_classes = num_classes

        self.embedding = EventNet(patch_size=patch_size, in_chans=in_chans, d_embed=d_model)
        self.model = RWKV(d_model=d_model, depth=depth, d_ffn=d_ffn, num_classes=num_classes)

    def forward(self, input):
        # input.shape = [batch_size, ctx_len, num_events]
        # output.shape = [batch_size, num_classes]

        x = self.embedding(input)    # [batch_size, ctx_len, d_model]
        output = self.model(x)       # [batch_size, num_classes]

        return output