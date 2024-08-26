import torch.nn as nn
from timm.models.registry import register_model
from models.encoder.rwkv_encoder import RWKVEncoder
from models.decoder.vit_decoder import ViTDecoder

class PretrainModel(nn.Module):

    def __init__(self,
                 patch_size=16,
                 ctx_len=512,
                 num_patches=4,
                 encoder_in_chans=3,
                 encoder_d_model=256,
                 encoder_depth=12,
                 encoder_d_ffn=1024,
                 encoder_num_classes=256,
                 decoder_d_model=256,
                 decoder_depth=12,
                 decoder_num_heads=8,
                 decoder_out_chans=2,
                 decoder_num_classes=512,
                 decoder_drop_rate=0., 
                 decoder_attn_drop_rate=0., 
                 decoder_norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.patch_size = patch_size
        self.ctx_len = ctx_len
        self.num_patches = num_patches
        self.encoder_in_chans = encoder_in_chans
        self.encoder_d_model = encoder_d_model
        self.encoder_depth = encoder_depth
        self.encoder_d_ffn = encoder_d_ffn
        self.encoder_num_classes = encoder_num_classes
        self.decoder_d_model = decoder_d_model
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_out_chans = decoder_out_chans
        self.decoder_num_classes = decoder_num_classes
        self.decoder_drop_rate = decoder_drop_rate
        self.decoder_attn_drop_rate = decoder_attn_drop_rate
        self.decoder_norm_layer = decoder_norm_layer

        self.encoder = RWKVEncoder(
            patch_size=patch_size,
            in_chans=encoder_in_chans, 
            d_model=encoder_d_model, 
            depth=encoder_depth, 
            d_ffn=encoder_d_ffn, 
            num_classes=encoder_num_classes,)
        self.decoder = ViTDecoder(
            patch_size=patch_size, 
            num_patches=num_patches, 
            d_model=decoder_d_model, 
            depth=decoder_depth, 
            num_heads=decoder_num_heads, 
            out_chans=decoder_out_chans, 
            num_classes=decoder_num_classes, 
            drop_rate=decoder_drop_rate, 
            attn_drop_rate=decoder_attn_drop_rate, 
            norm_layer=decoder_norm_layer)

    def forward(self, input):
        # input.shape = [batch_size, ctx_len, num_patches, num_events]
        # output.shape = [batch_size, num_patches, num_classes]

        B, T, P, C = input.size()
        x = input.permute(0, 2, 1, 3).reshape(-1, T, C)  # [batch_size * num_patches, ctx_len, num_events]

        x = self.encoder(x)                              # [batch_size * num_patches, d_model]
        x = x.reshape(B, P, -1)                          # [batch_size, num_patches, d_model]
        output = self.decoder(x)                         # [batch_size, num_patches, num_classes]

        return output