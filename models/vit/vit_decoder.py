import torch
import torch.nn as nn
from .vit_layers import Block, get_sinusoid_encoding_table 

class ViTDecoder(nn.Module):
    def __init__(
        self,
        in_chans: int,
        npatches: int,
        d_model: int,
        depth: int,
        nheads: int,
        num_classes: int,
        drop_rate: float,
        attn_drop_rate: float,
    ):
        super().__init__()


        self.in_chans = in_chans
        self.npatches = npatches
        self.d_model = d_model
        self.depth = depth
        self.num_heads = nheads
        self.num_classes = num_classes
        
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = nn.LayerNorm
        
        self.embedding = nn.Linear(in_chans, d_model)
        self.pos_embed = get_sinusoid_encoding_table(npatches, d_model).cuda()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=nheads, drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer)
            for _ in range(depth)])
        
        self.norm = self.norm_layer(d_model)
        self.head = nn.Linear(d_model, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, input):
        # input.shape = [batch_size, n_patches, d_model]
        # output.shape = [batch_size, num_classes]
        
        batch_size, _, _ = input.shape
        x = self.embedding(input)   # [batch_size, n_patches, d_model]
        x = x + self.pos_embed      # [batch_size, n_patches, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)   # [batch_size, n_patches + 1, d_model]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x[:, 0]
        output = self.head(x)
        return output