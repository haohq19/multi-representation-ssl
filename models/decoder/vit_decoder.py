import torch.nn as nn
from .vit_layers import Block, get_sinusoid_encoding_table 

class ViTDecoder(nn.Module):
    def __init__(self, 
                 patch_size=16,
                 num_patches=4,
                 d_model=768,
                 depth=12,
                 num_heads=12, 
                 out_chans=2,
                 num_classes=512,
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads

        self.out_chans = out_chans
        assert num_classes == out_chans * patch_size ** 2
        self.num_classes = num_classes
        
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        
        self.pos_embed = get_sinusoid_encoding_table(num_patches, d_model)

        self.blocks = nn.ModuleList([
            Block(d_model=d_model, num_heads=num_heads, drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(depth)])
        
        self.norm = norm_layer(d_model)
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
        # input.shape = [batch_size, num_patches, d_model]
        # output.shape = [batch_size, num_patches, num_classes]
        
        B, _, _ = input.size()
        if self.pos_embed is not None:
            x = input + self.pos_embed.expand(B, -1, -1).type_as(input).to(input.device).clone().detach()
        else:
            x = input

        for blk in self.blocks:
            x = blk(x)

        output = self.head(self.norm(x))
        return output