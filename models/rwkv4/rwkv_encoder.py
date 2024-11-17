import torch.nn as nn
from .rwkv import RWKV
from ..tokenizer.event_tokenizer import EventTokenizer

class RWKVEncoder(nn.Module):

    def __init__(self, patch_size, in_chans, d_model, depth, d_ffn, num_classes):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.d_model = d_model
        self.depth = depth
        self.d_ffn = d_ffn
        self.num_classes = num_classes

        self.embedding = EventTokenizer(patch_size=patch_size, d_embed=d_model)
        self.model = RWKV(d_model=d_model, depth=depth, d_ffn=d_ffn, num_classes=num_classes)

    def forward(self, input):
        # input.shape = [batch_size, n_events, 4]
        # output.shape = [batch_size, num_classes]

        x = self.embedding(input)   # [batch_size, n_events, d_model]
        output = self.model(x)      # [batch_size, num_classes]

        return output