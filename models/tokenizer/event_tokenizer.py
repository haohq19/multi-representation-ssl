import torch
import torch.nn as nn
from .temporal_embedding import SinusoidalEmbedding

class EventTokenizer(nn.Module):
    def __init__(self, patch_size, d_embed):
        super().__init__()
        self.patch_size = patch_size
        self.vocab_size = 2 * patch_size * patch_size
        self.d_embed = d_embed

        self.embedding = nn.Embedding(self.vocab_size, d_embed)
        self.ln = nn.LayerNorm(d_embed)
        self.temporal_embedding = SinusoidalEmbedding(d_embed)

    def forward(self, input):
        """
        Args:
            input: torch.Tensor, shape [batch_size, n_events, 4]
        Returns:
            output: torch.Tensor, shape [batch_size, n_events, d_embed]
        """
        timestamps = input[:, :, 0]
        event_ids = input[:, :, 1] * self.patch_size + input[:, :, 2] + input[:, :, 3] * self.patch_size * self.patch_size
        # event_ids = input[:, :, 1] * self.patch_size + input[:, :, 2] * 2 + input[:, :, 3]    # S7 paper implementation
        x = self.embedding(event_ids)
        x = self.ln(x)
        timestamps_embed = self.temporal_embedding(timestamps)
        output = x + timestamps_embed
        return output

