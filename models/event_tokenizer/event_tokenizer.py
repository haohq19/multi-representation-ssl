import torch.nn as nn
from .temporal_embedding import SinusoidalEmbedding

class EventTokenizer(nn.Module):
    def __init__(self, patch_size, d_embed):
        super().__init__()
        self.patch_size = patch_size
        self.d_embed = d_embed
        self.vocab_size = 2 * self.patch_size * self.patch_size
        

        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.ln = nn.LayerNorm(self.d_embed)
        self.temporal_embedding = SinusoidalEmbedding(self.d_embed)

    def forward(self, input):
        """
        Args:
            input: torch.Tensor, shape [batch_size, n_events, 4]
        Returns:
            output: torch.Tensor, shape [batch_size, n_events, d_embed]
        """
        timestamps = input[:, :, 0].float()
        x = input[:, :, 1].long()
        y = input[:, :, 2].long()
        p = input[:, :, 3].long()
        event_ids = p * self.patch_size + y * self.patch_size + x
        x = self.embedding(event_ids)
        x = self.ln(x)
        timestamps_embed = self.temporal_embedding(timestamps)
        output = x + timestamps_embed
        return output

