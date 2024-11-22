import torch
import torch.nn as nn

# implemente a temporal embedding module
class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, timestamps):
        """
        Args:
            timestamps: torch.Tensor, shape [batch_size, n_events]
        Returns:
            output: torch.Tensor, shape [batch_size, n_events, d_embed]
        """
        batch_size, n_events = timestamps.shape
        d_embed = self.d_embed
        
        # convert timestamps to difference
        diff = timestamps[:, 1:] - timestamps[:, :-1]
        diff = torch.cat([torch.zeros(batch_size, 1).to(diff.device), diff], dim=1)

        # calculate the sinusoidal embedding
        position = torch.arange(0, d_embed, 2).float().to(diff.device)
        div_term = torch.exp(position * -(torch.log(torch.tensor(10000.0)) / d_embed))
        emb = torch.zeros(batch_size, n_events, d_embed).to(diff.device)
        emb[:, :, 0::2] = torch.sin(diff.unsqueeze(-1) * div_term)
        emb[:, :, 1::2] = torch.cos(diff.unsqueeze(-1) * div_term)

        return emb